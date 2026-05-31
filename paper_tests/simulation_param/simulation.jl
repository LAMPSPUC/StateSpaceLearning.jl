using Pkg
Pkg.activate(".")
Pkg.instantiate()
using StateSpaceLearning
using PyCall

using Random, Statistics, DataFrames, LinearAlgebra
using CSV
using Distributions
using Dates
using Printf

include("simulation_generator.jl")
include("metrics.jl")
include("evaluate_models.jl")

function format_time_diff(diff::Millisecond)
    """Format a time difference in HH:MM:SS"""
    total_seconds = div(diff.value, 1000)
    hours = div(total_seconds, 3600)
    minutes = div(rem(total_seconds, 3600), 60)
    seconds = rem(total_seconds, 60)
    return @sprintf("%02d:%02d:%02d", hours, minutes, seconds)
end

function kalman_components_statespacemodels(y::AbstractVector{<:AbstractFloat}, s::Int)
    py"""
    import math
    import statsmodels.api as sm
    import numpy as np
    def evaluate_ss(y_train, s):
        model_components = {'irregular': True, 'level': True, 'trend': True, 'seasonal': s, 
                                    'stochastic_level': True, 'stochastic_trend': True, 'stochastic_seasonal': True}
        model = sm.tsa.UnobservedComponents(np.array(y_train), **model_components)
        results = model.fit(disp=False, maxiter=1e5)

        estimated_innovations = {
            "level": results.level["smoothed"],
            "trend": results.trend["smoothed"],
            "seasonal": results.seasonal["smoothed"],
        }
        return estimated_innovations
    """

    estimated_innovations = py"evaluate_ss"(y, s)

    μ_hat = estimated_innovations["level"]
    ν_hat = estimated_innovations["trend"]
    γ_hat = estimated_innovations["seasonal"]

    return μ_hat, ν_hat, γ_hat
end

function component_metrics(
    μ_1::AbstractVector,
    ν_1::AbstractVector,
    γ_1::AbstractVector,
    μ_2::AbstractVector,
    ν_2::AbstractVector,
    γ_2::AbstractVector,
    method::AbstractString,
    sample_size::Int,
    replicate::Int,
)
    return DataFrame(;
        sample_size=fill(sample_size, 3),
        replicate=fill(replicate, 3),
        method=fill(method, 3),
        component=["level", "slope", "seasonal"],
        bias=[bias_func(μ_1, μ_2), bias_func(ν_1, ν_2), bias_func(γ_1, γ_2)],
    )
end

function run_experiment(sample_sizes::Vector{Int}; reps::Int=50, compare::AbstractString="both")
    results = DataFrame()
    start_time = now()
    total_iterations = length(sample_sizes) * reps
    current_iteration = 0

    for (idx, T) in enumerate(sample_sizes)
        @info "Running experiment with sample size: $T and $reps repetitions"
        for rep in 1:reps
            current_iteration += 1
            
            s = 12
            y, μ, ν, γ_vec, xi_std, zeta_std, omega_std, eps_std = generate_series(T, rep)

            μ_ssl, ν_ssl, γ_ssl = get_SSL_results(y, s, μ, ν, γ_vec, "aic")

            # Compute Kalman components only if needed (to save time)
            μ_kal = ν_kal = γ_kal = nothing
            need_kalman = lowercase(compare) in ("kalman", "both")
            if need_kalman
                μ_kal, ν_kal, γ_kal = kalman_components_statespacemodels(y, s)
            end

            # Collect comparisons based on `compare` parameter
            if lowercase(compare) in ("true", "both")
                ssl_vs_true_df = component_metrics(
                    μ_ssl, ν_ssl, γ_ssl, μ, ν, γ_vec, "SSL vs True", T, rep
                )
                results = vcat(results, ssl_vs_true_df)
            end

            if need_kalman
                ssl_vs_kalman_df = component_metrics(
                    μ_ssl, ν_ssl, γ_ssl, μ_kal, ν_kal, γ_kal, "SSL vs Kalman", T, rep
                )
                results = vcat(results, ssl_vs_kalman_df)
            end
            
            # Progress logging
            elapsed = now() - start_time
            elapsed_ms = elapsed.value
            avg_time_per_iter_ms = elapsed_ms / current_iteration
            remaining_iterations = total_iterations - current_iteration
            estimated_remaining_ms = avg_time_per_iter_ms * remaining_iterations
            estimated_total_ms = elapsed_ms + estimated_remaining_ms
            
            @printf("\n[Progress] Iteration %d/%d (%.1f%%)\n", current_iteration, total_iterations, 100*current_iteration/total_iterations)
            @printf("  Sample size: %d | Replicate: %d\n", T, rep)
            @printf("  Elapsed: %s\n", format_time_diff(Millisecond(round(Int, elapsed_ms))))
            @printf("  Estimated remaining: %s\n", format_time_diff(Millisecond(round(Int, estimated_remaining_ms))))
            @printf("  Estimated total time: %s\n", format_time_diff(Millisecond(round(Int, estimated_total_ms))))
        end
    end

    total_elapsed = now() - start_time
    @printf("\n✓ Experiment completed in %s\n", format_time_diff(total_elapsed))
    return results
end

function paired_significance(results::DataFrame)
    alpha = 0.05
    stats_rows = DataFrame()

    for group in groupby(results, [:sample_size, :component, :method])
        sample_size = group.sample_size[1]
        component = group.component[1]
        method = group.method[1]

        col_bias = group[:, :bias]

        n = length(col_bias)
        if n <= 1
            continue
        end

        mean_bias = mean(col_bias)
        std_bias = std(col_bias; corrected=true)
        stderr = std_bias / sqrt(n)

        # One-sample t-test: H0: bias = 0
        if stderr > 0
            t_stat = mean_bias / stderr
            p_value = 2 * (1 - cdf(TDist(n - 1), abs(t_stat)))
        else
            t_stat = 0.0
            p_value = 1.0
        end
        significant = p_value < alpha

        stats_rows = vcat(
            stats_rows,
            DataFrame(;
                sample_size=sample_size,
                component=component,
                method=method,
                mean_bias=mean_bias,
                std_bias=std_bias,
                stderr=stderr,
                t_stat=t_stat,
                p_value=p_value,
                significant=significant,
                alpha=alpha,
                n=n,
            ),
        )
    end

    return stats_rows
end

default_sample_sizes = [60, 120, 240, 480, 960]
reps = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1000
sample_sizes = length(ARGS) >= 2 ? parse.(Int, split(ARGS[2], ",")) : default_sample_sizes
compare_arg = length(ARGS) >= 3 ? ARGS[3] : "both"

results = run_experiment(sample_sizes; reps=reps, compare=compare_arg)

paired_stats = paired_significance(results)

CSV.write("paper_tests/simulation_param/ssl_paired_tests_by_method.csv", paired_stats)

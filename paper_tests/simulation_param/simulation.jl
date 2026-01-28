using Pkg
Pkg.activate(".")
Pkg.instantiate()
using StateSpaceLearning
using PyCall

using Random, Statistics, DataFrames, LinearAlgebra
using CSV
using Distributions

include("simulation_generator.jl")
include("metrics.jl")
include("evaluate_models.jl")

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

function run_experiment(sample_sizes::Vector{Int}; reps::Int=50)
    results = DataFrame()

    for (idx, T) in enumerate(sample_sizes)
        @info "Running experiment with sample size: $T and $reps repetitions"
        for rep in 1:reps
            s = 12
            y, μ, ν, γ_vec, xi_std, zeta_std, omega_std, eps_std = generate_series(T, rep)

            μ_ssl, ν_ssl, γ_ssl = get_SSL_results(y, s, μ, ν, γ_vec, "aic")

            μ_kal, ν_kal, γ_kal = kalman_components_statespacemodels(y, s)
            align_components!(μ_kal, ν_kal, γ_kal, μ, ν, γ_vec)

            ssl_true_df = component_metrics(
                μ_ssl, ν_ssl, γ_ssl, μ, ν, γ_vec, "SSL vs True", T, rep
            )
            kal_true_df = component_metrics(
                μ_kal, ν_kal, γ_kal, μ, ν, γ_vec, "Kalman vs True", T, rep
            )

            results = vcat(results, ssl_true_df)
            results = vcat(results, kal_true_df)
        end
    end

    return results
end

function paired_significance(results::DataFrame)
    alpha = 0.05

    ssl_true = filter(row -> row.method == "SSL vs True", results)
    kal_true = filter(row -> row.method == "Kalman vs True", results)

    stats_rows = DataFrame()

    for group_ssl in groupby(ssl_true, [:sample_size, :component])
        sample_size = group_ssl.sample_size[1]
        component = group_ssl.component[1]

        group_kal = filter(
            row -> row.sample_size == sample_size && row.component == component, kal_true
        )

        if isempty(group_kal)
            @warn "No matching Kalman group found for $sample_size, $component"
            continue
        end

        col_ssl = group_ssl[:, :bias]
        col_kal = group_kal[:, :bias]

        paired_diffs = col_ssl .- col_kal

        n = length(paired_diffs)
        if n <= 1
            continue
        end

        mean_diff = mean(paired_diffs)
        std_diff = std(paired_diffs; corrected=true)
        stderr = std_diff / sqrt(n)

        if stderr > 0
            t_stat = mean_diff / stderr
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
                mean_bias_diff=mean_diff,
                std_bias_diff=std_diff,
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
reps = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 50
sample_sizes = length(ARGS) >= 2 ? parse.(Int, split(ARGS[2], ",")) : default_sample_sizes

results = run_experiment(sample_sizes; reps=reps)

paired_stats = paired_significance(results)

CSV.write("paper_tests/simulation_param/ssl_vs_kalman_paired_tests.csv", paired_stats)

gg = [
    ["Hourly", "H"],
    ["Daily", "D"],
    ["Weekly", "W"],
    ["Monthly", "M"],
    ["Quarterly", "Q"],
    ["Yearly", "Y"],
]

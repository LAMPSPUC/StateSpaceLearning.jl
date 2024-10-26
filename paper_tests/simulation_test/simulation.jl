using Pkg: Pkg
Pkg.activate(".")
Pkg.instantiate()
using StateSpaceLearning

Pkg.activate("")
using CSV, DataFrames, StateSpaceModels, Statistics, GLM
using PyCall
using GLMNet

include("metrics.jl")
include("evaluate_models.jl")
include("simulation_generator.jl")

using Distributed
using Pkg: Pkg
Pkg.activate(".")
Pkg.instantiate()
using StateSpaceLearning

nprocs = parse(Int, ARGS[1])
nprocs > 1 ? addprocs(nprocs) : nothing
@everywhere begin
    using Pkg: Pkg
    using CSV, DataFrames, StateSpaceModels, Statistics, PyCall, Distributions, GLMNet,
          Polynomials
end

@everywhere begin
    using Pkg: Pkg
    Pkg.activate(".")
    using StateSpaceLearning
end

@everywhere begin
    include("metrics.jl")
    include("evaluate_models.jl")
    include("simulation_generator.jl")
end

function create_dirs()
    try
        mkdir("paper_tests/simulation_test/results_simulation_raw")
    catch
        @warn "Directory already exists"
    end
    try
        mkdir("paper_tests/simulation_test/results_simulation")
    catch
        @warn "Directory already exists"
    end
end

create_dirs()

@everywhere begin
    function get_M_K_res(M_K_d)
        M = M_K_d["M"]
        K = M_K_d["K"]
        N = M_K_d["N"]
        rr1 = DataFrame()
        rr2 = DataFrame()
        rr3 = DataFrame()
        rr4 = DataFrame()
        rr5 = DataFrame()
        rr6 = DataFrame()
        for i in 1:50
            y_featured, true_exps, X, true_β = generate_subset(60, M, K)
            y_featured = M_K_d["p_vec_M_K"][i]["y_featured"]
            true_exps = M_K_d["p_vec_M_K"][i]["true_exps"]
            X = M_K_d["p_vec_M_K"][i]["X"]
            true_β = M_K_d["p_vec_M_K"][i]["true_β"]
            @info(i)
            series_result1 = get_SSL_results(y_featured, collect(1:K),
                                             setdiff(collect(1:M), collect(1:K)), X, "aic",
                                             true_β)
            series_result2 = get_SSL_results(y_featured, collect(1:K),
                                             setdiff(collect(1:M), collect(1:K)), X, "bic",
                                             true_β)
            series_result3, converged3 = get_forward_ss(y_featured, collect(1:K),
                                                        setdiff(collect(1:M), collect(1:K)),
                                                        X, "aic", true_β)
            series_result4, converged4 = get_forward_ss(y_featured, collect(1:K),
                                                        setdiff(collect(1:M), collect(1:K)),
                                                        X, "bic", true_β)
            series_result5, converged5 = get_SS_res_results(y_featured, collect(1:K),
                                                            setdiff(collect(1:M),
                                                                    collect(1:K)), X, "aic",
                                                            true_β)
            series_result6, converged6 = get_SS_res_results(y_featured, collect(1:K),
                                                            setdiff(collect(1:M),
                                                                    collect(1:K)), X, "bic",
                                                            true_β)
            rr1 = vcat(rr1, series_result1)
            rr2 = vcat(rr2, series_result2)
            rr3 = vcat(rr3, series_result3)
            rr4 = vcat(rr4, series_result4)
            rr5 = vcat(rr5, series_result5)
            rr6 = vcat(rr6, series_result6)
            i += 1
            @info("K: $K, M: $M")
            @info(mean(rr2[!, "true_positives"]))
            @info(mean(rr2[!, "false_positives"]))
        end
        CSV.write("paper_tests/simulation_test/results_simulation_raw/SSL_aic_$(M)$(K)_$N.csv",
                  rr1)
        CSV.write("paper_tests/simulation_test/results_simulation_raw/SSL_bic_$(M)$(K)_$N.csv",
                  rr2)
        CSV.write("paper_tests/simulation_test/results_simulation_raw/SS_aic_f$(M)$(K)_$N.csv",
                  rr3)
        CSV.write("paper_tests/simulation_test/results_simulation_raw/SS_bic_f$(M)$(K)_$N.csv",
                  rr4)
        CSV.write("paper_tests/simulation_test/results_simulation_raw/SS_aic_$(M)$(K)_$N.csv",
                  rr5)
        return CSV.write("paper_tests/simulation_test/results_simulation_raw/SS_bic_$(M)$(K)_$N.csv",
                         rr6)
    end
end

Random.seed!(2024)
p_vec = []
for M in [50, 100]
    for K in [3, 5, 8, 10]
        for N in 1:10
            p_vec_M_K = Dict()
            for i in 1:50
                p_vec_M_K[i] = Dict()
                y_featured, true_exps, X, true_β = generate_subset(60, M, K)
                p_vec_M_K[i]["y_featured"] = y_featured
                p_vec_M_K[i]["true_exps"] = true_exps
                p_vec_M_K[i]["X"] = X
                p_vec_M_K[i]["true_β"] = true_β
            end
            push!(p_vec, Dict("M" => M, "K" => K, "N" => N, "p_vec_M_K" => p_vec_M_K))
        end
    end
end

function robust_pmap(f::Function, a; num_retries::Int=3)
    return pmap(f, a; on_error=e -> rethrow(),
                retry_delays=ExponentialBackOff(; n=num_retries))
end

robust_pmap(get_M_K_res, p_vec)

for M in [50, 100]
    for K in [3, 5, 8, 10]
        last_df1 = DataFrame()
        last_df2 = DataFrame()
        last_df3 = DataFrame()
        last_df4 = DataFrame()
        last_df5 = DataFrame()
        last_df6 = DataFrame()
        for i in 1:10
            df1 = CSV.read("paper_tests/simulation_test/results_simulation_raw/SSL_aic_$(M)$(K)_$i.csv",
                           DataFrame)
            df2 = CSV.read("paper_tests/simulation_test/results_simulation_raw/SSL_bic_$(M)$(K)_$i.csv",
                           DataFrame)
            df3 = CSV.read("paper_tests/simulation_test/results_simulation_raw/SS_aic_$(M)$(K)_$i.csv",
                           DataFrame)
            df4 = CSV.read("paper_tests/simulation_test/results_simulation_raw/SS_bic_$(M)$(K)_$i.csv",
                           DataFrame)
            df5 = CSV.read("paper_tests/simulation_test/results_simulation_raw/SS_aic_f$(M)$(K)_$i.csv",
                           DataFrame)
            df6 = CSV.read("paper_tests/simulation_test/results_simulation_raw/SS_bic_f$(M)$(K)_$i.csv",
                           DataFrame)
            last_df1 = vcat(last_df1, df1)
            last_df2 = vcat(last_df2, df2)
            last_df3 = vcat(last_df3, df3)
            last_df4 = vcat(last_df4, df4)
            last_df5 = vcat(last_df5, df5)
            last_df6 = vcat(last_df6, df6)
        end
        CSV.write("paper_tests/simulation_test/results_simulation/SSL_aic_$(M)$(K).csv",
                  last_df1)
        CSV.write("paper_tests/simulation_test/results_simulation/SSL_bic_$(M)$(K).csv",
                  last_df2)
        CSV.write("paper_tests/simulation_test/results_simulation/SS_aic_$(M)$(K).csv",
                  last_df3)
        CSV.write("paper_tests/simulation_test/results_simulation/SS_bic_$(M)$(K).csv",
                  last_df4)
        CSV.write("paper_tests/simulation_test/results_simulation/SS_aic_f$(M)$(K).csv",
                  last_df5)
        CSV.write("paper_tests/simulation_test/results_simulation/SS_bic_f$(M)$(K).csv",
                  last_df6)
    end
end

function get_metrics(df, q, n)
    T = size(df, 1)
    true_model_rate = 0
    all_true_positives_rate = 0
    true_positive_rate = 0
    true_negative_rate = 0
    positive_rate = 0
    false_positive_rate = 0
    for i in 1:T
        i_df = df[i, :]
        if i_df["true_positives"] == q && i_df["true_negatives"] == n - q &&
           i_df["false_positives"] == 0 && i_df["false_negatives"] == 0
            true_model_rate += 1 / T
        end
        if i_df["true_positives"] == q
            all_true_positives_rate += 1 / T
        end
        true_positive_rate += i_df["true_positives"] / (T * q)
        true_negative_rate += i_df["true_negatives"] / (T * (n - q))
        positive_rate += (i_df["true_positives"] + i_df["false_positives"]) / T
        false_positive_rate += (i_df["false_positives"]) / T
    end
    return true_model_rate, all_true_positives_rate, true_positive_rate, true_negative_rate,
           positive_rate, false_positive_rate
end

df_dict = Dict()
df_dict["SS_aic"] = Dict()
df_dict["SS_bic"] = Dict()
df_dict["SS_f_aic"] = Dict()
df_dict["SS_f_bic"] = Dict()
df_dict["SSL_aic"] = Dict()
df_dict["SSL_bic"] = Dict()

for n in [50, 100]
    for q in [3, 5, 8, 10]
        df_SS_aic = CSV.read("paper_tests/simulation_test/results_simulation/SS_aic_$n$(q).csv",
                             DataFrame)
        df_SS_bic = CSV.read("paper_tests/simulation_test/results_simulation/SS_bic_$n$(q).csv",
                             DataFrame)
        df_SS_aicf = CSV.read("paper_tests/simulation_test/results_simulation/SS_aic_f$n$(q).csv",
                              DataFrame)
        df_SS_bicf = CSV.read("paper_tests/simulation_test/results_simulation/SS_bic_f$n$(q).csv",
                              DataFrame)
        df_SSL_aic = CSV.read("paper_tests/simulation_test/results_simulation/SSL_aic_$n$(q).csv",
                              DataFrame)
        df_SSL_bic = CSV.read("paper_tests/simulation_test/results_simulation/SSL_bic_$n$(q).csv",
                              DataFrame)
        df_dict["SS_aic"][n, q] = get_metrics(df_SS_aic, q, n)
        df_dict["SS_bic"][n, q] = get_metrics(df_SS_bic, q, n)
        df_dict["SS_f_aic"][n, q] = get_metrics(df_SS_aicf, q, n)
        df_dict["SS_f_bic"][n, q] = get_metrics(df_SS_bicf, q, n)
        df_dict["SSL_aic"][n, q] = get_metrics(df_SSL_aic, q, n)
        df_dict["SSL_bic"][n, q] = get_metrics(df_SSL_bic, q, n)
    end
end

df = DataFrame()

for name in ["SSL_aic", "SSL_bic", "SS_aic", "SS_bic", "SS_f_aic", "SS_f_bic"]
    for n in [50, 100]
        column = []
        for i in 1:6
            for q in [3, 5, 8, 10]
                push!(column, round(df_dict[name][n, q][i]; digits=3))
            end
        end
        df[!, Symbol(name * "_" * string(n))] = column
    end
end

CSV.write("paper_tests/simulation_test/results_metrics/metrics_confusion_matrix.csv", df)

df_mse_bias = DataFrame()

function convert_to_sci_notation(num::Float64)
    # Get the exponent part of the number in scientific notation
    exp_part = floor(log10(abs(num)))

    # Divide the number by 10^(exp_part) to get the mantissa
    mantissa = num / 10^(exp_part)

    # Round the mantissa to have one decimal place
    rounded_mantissa = round(mantissa; digits=1)

    # Construct the string representation of the result in scientific notation
    result_str = string(rounded_mantissa, "e", exp_part)

    return result_str[1:(end - 2)]
end

for name in ["SSL_aic_", "SSL_bic_", "SS_aic_", "SS_bic_", "SS_aic_f", "SS_bic_f"]
    for n in [50, 100]
        column = []
        for i in 1:5
            for q in [3, 5, 8, 10]
                df_name = CSV.read("paper_tests/simulation_test/results_simulation/$(name)$n$(q).csv",
                                   DataFrame)
                if i == 1
                    num = round(mean(df_name[:, "mse"]); digits=3)
                    if num > 10
                        push!(column, convert_to_sci_notation(num))
                    else
                        push!(column, num)
                    end
                elseif i == 2
                    num = round(mean(df_name[:, "bias"]); digits=3)
                    if num > 10
                        push!(column, convert_to_sci_notation(num))
                    else
                        push!(column, num)
                    end
                elseif i == 3
                    num = round(mean(df_name[:, "time"]); digits=3)
                    if num > 10
                        push!(column, convert_to_sci_notation(num))
                    else
                        push!(column, num)
                    end
                elseif i == 4
                    mse_vec = df_name[:, "mse"]
                    q25 = quantile(mse_vec, 0.25)
                    q75 = quantile(mse_vec, 0.75)
                    IQR = q75 - q25
                    #exclude out of iqr values
                    mse_vec = mse_vec[(q25 - 1.5 * IQR .<= mse_vec) .& (mse_vec .<= q75 + 1.5 * IQR)]
                    num = round(mean(mse_vec); digits=3)
                    if num > 10
                        push!(column, convert_to_sci_notation(num))
                    else
                        push!(column, num)
                    end
                else
                    bias_vec = df_name[:, "bias"]
                    q25 = quantile(bias_vec, 0.25)
                    q75 = quantile(bias_vec, 0.75)
                    IQR = q75 - q25
                    #exclude out of iqr values
                    bias_vec = bias_vec[(q25 - 1.5 * IQR .<= bias_vec) .& (bias_vec .<= q75 + 1.5 * IQR)]
                    num = round(mean(bias_vec); digits=3)
                    if num > 10
                        push!(column, convert_to_sci_notation(num))
                    else
                        push!(column, num)
                    end
                end
            end
        end
        df_mse_bias[!, Symbol(name * "_" * string(n))] = column
    end
end
CSV.write("paper_tests/simulation_test/results_simulation/metrics_bias_mse.csv",
          df_mse_bias)

using Pkg: Pkg
Pkg.activate(".")
Pkg.instantiate()

using StateSpaceLearning, CSV, DataFrames, Statistics, Revise, HTTP, PyCall

base_url = "https://raw.githubusercontent.com/Mcompetitions/M4-methods/refs/heads/master/Dataset"

# Helper function to read CSV from URL
function read_csv_from_url(url::String)
    response = HTTP.get(url)
    return CSV.read(IOBuffer(response.body), DataFrame)
end

df_train = read_csv_from_url("$(base_url)/Train/Monthly-train.csv")
df_test = read_csv_from_url("$(base_url)/Test/Monthly-test.csv")

df_train_daily = read_csv_from_url("$(base_url)/Train/Daily-train.csv")
df_test_daily = read_csv_from_url("$(base_url)/Test/Daily-test.csv")

df_train_quarterly = read_csv_from_url("$(base_url)/Train/Quarterly-train.csv")
df_test_quarterly = read_csv_from_url("$(base_url)/Test/Quarterly-test.csv")

df_train_hourly = read_csv_from_url("$(base_url)/Train/Hourly-train.csv")
df_test_hourly = read_csv_from_url("$(base_url)/Test/Hourly-test.csv")

df_train_weekly = read_csv_from_url("$(base_url)/Train/Weekly-train.csv")
df_test_weekly = read_csv_from_url("$(base_url)/Test/Weekly-test.csv")

df_train_yearly = read_csv_from_url("$(base_url)/Train/Yearly-train.csv")
df_test_yearly = read_csv_from_url("$(base_url)/Test/Yearly-test.csv")

include("metrics.jl")
include("evaluate_model.jl")
include("prepare_data.jl")

complete_dict_vec = Dict("M" => build_train_test_dict(df_train, df_test;name="M"),
                         "D" => build_train_test_dict(df_train_daily, df_test_daily;name="D"),
                         "Q" => build_train_test_dict(df_train_quarterly, df_test_quarterly;name="Q"),
                         "H" => build_train_test_dict(df_train_hourly, df_test_hourly;name="H"),
                         "W" => build_train_test_dict(df_train_weekly, df_test_weekly;name="W"),
                         "Y" => build_train_test_dict(df_train_yearly, df_test_yearly;name="Y"),
)

parameters_dict = Dict(
    "M" => Dict("freq_seasonal" => 12, "cycle_period" => 0, "m" => 12, "ξ_threshold" => 1, "ζ_threshold" => 12, "ω_threshold" => 12, "ϕ_threshold" => 0, "seasonal" => "stochastic", "cycle" => "none", "sample_size" => 60, "name" => "MONTH", "H" => 18, "NAIVE_sMAPE" => 14.427, "NAIVE_MASE" => 1.063),
    "Q" => Dict("freq_seasonal" => 4, "cycle_period" => 0, "m" => 4, "ξ_threshold" => 1, "ζ_threshold" => 4, "ω_threshold" => 4, "ϕ_threshold" => 0, "seasonal" => "stochastic", "cycle" => "none", "sample_size" => "all", "name" => "QUARTERLY", "H" => 8, "NAIVE_sMAPE" => 11.012, "NAIVE_MASE" => 1.371),
    "D" => Dict("freq_seasonal" => 1, "cycle_period" => 0, "m" => 1, "ξ_threshold" => 1, "ζ_threshold" => 7, "ω_threshold" => 1, "ϕ_threshold" => 0, "seasonal" => "none", "cycle" => "none", "sample_size" => 90, "name" => "DAILY", "H" => 14, "NAIVE_sMAPE" => 3.045, "NAIVE_MASE" => 3.278),
    "W" => Dict("freq_seasonal" => 1, "cycle_period" => 0, "m" => 1, "ξ_threshold" => 1, "ζ_threshold" => 4, "ω_threshold" => 1, "ϕ_threshold" => 0, "seasonal" => "none", "cycle" => "none", "sample_size" => 104, "name" => "WEEKLY", "H" => 13, "NAIVE_sMAPE" => 9.161, "NAIVE_MASE" => 2.777),
    "Y" => Dict("freq_seasonal" => 1, "cycle_period" => 0, "m" => 1, "ξ_threshold" => 1, "ζ_threshold" => 2, "ω_threshold" => 1, "ϕ_threshold" => 0, "seasonal" => "none", "cycle" => "none", "sample_size" => "all", "name" => "YEARLY", "H" => 6, "NAIVE_sMAPE" => 16.342, "NAIVE_MASE" => 3.974),
    "H" => Dict("freq_seasonal" => 168, "cycle_period" => [24], "m" => 24, "ξ_threshold" => 1, "ζ_threshold" => 168, "ω_threshold" => 168, "ϕ_threshold" => 12, "seasonal" => "stochastic", "cycle" => "stochastic", "sample_size" => 720, "name" => "HOURLY", "H" => 48, "NAIVE_sMAPE" => 18.383, "NAIVE_MASE" => 2.395),
)

# Function to append results to CSV file
function append_results(filepath, results_df)
    if isfile(filepath)
        df_old = CSV.read(filepath, DataFrame)
        results_df = vcat(df_old, results_df)
        @info("Average MASE: $(mean(results_df[:, 1]))")
        @info("Average sMAPE: $(mean(results_df[:, 2]))")
        @info("Average CRPS: $(mean(results_df[:, 3]))")
    end
    return CSV.write(filepath, results_df)
end

function run_config(
    dict_vec::Vector,
    results_table::DataFrame,
    outlier::Bool,
    selection::String,
    information_criteria::String,
    α::AbstractFloat,
    param::Dict
)
    results_df = DataFrame()
    initialization_df = DataFrame()
    filepath = "paper_tests/m4_test/results_SSL/$(param["name"])/$(information_criteria)_$(α)_$(outlier).csv"
    CSV.write(filepath, results_df)

    clear_df_number = 1000

    for i in eachindex(dict_vec)
        if i % clear_df_number == 1 # Clear DataFrame to save memory
            results_df = DataFrame()
            initialization_df = DataFrame()
        end

        initialization_df, results_df = evaluate_SSL(
            initialization_df,
            results_df,
            dict_vec[i],
            outlier,
            α,
            selection,
            information_criteria,
            param
        )

        if i % clear_df_number == 0 || i == length(dict_vec)
            @info "Saving results for $i series"
            append_results(filepath, results_df)
        end
    end

    results_df = CSV.read(filepath, DataFrame)

    mase = round(mean(results_df[:, :MASE]); digits=3)
    smape = round(mean(results_df[:, :sMAPE]); digits=3)
    owa = round(
        mean([
            mean(results_df[:, :sMAPE]) / param["NAIVE_sMAPE"],
            mean(results_df[:, :MASE]) / param["NAIVE_MASE"],
        ]);
        digits=3,
    )
    crps = round(mean(results_df[:, :CRPS]); digits=3)
    name = if outlier
        "SSL-O ($(information_criteria), α = $(α))"
    else
        "SSL ($(information_criteria), α = $(α))"
    end
    results_table = vcat(
        results_table,
        DataFrame(
            "Names" => ["$name"],
            "MASE" => [mase],
            "sMAPE" => [smape],
            "OWA" => [owa],
            "CRPS" => [crps],
        ),
    )
    return results_table
end

function run_benchmark_model(dict_vec::Vector, param::Dict)
    results_df = DataFrame()
    m = param["m"]
    H = param["H"]
    frequency = param["freq_seasonal"]
    filepath = "paper_tests/m4_test/metrics_results/BENCHMARK_$(param["name"])_RESULTS.csv"
    CSV.write(filepath, results_df)

    clear_df_number = 1000

    for i in eachindex(dict_vec)
        if i % clear_df_number == 1 # Clear DataFrame to save memory
            results_df = DataFrame()
        end

        ss_results = evaluate_SS(dict_vec[i], m, H, frequency)
        results_df = vcat(results_df, ss_results)

        if i % clear_df_number == 0 || i == length(dict_vec)
            @info "Saving results for $i series"
            append_results(filepath, results_df)
        end
    end

    results_df = CSV.read(filepath, DataFrame)

    mase = round(mean(results_df[:, :MASE]); digits=3)
    smape = round(mean(results_df[:, :sMAPE]); digits=3)
    owa = round(
        mean([
            mean(results_df[:, :sMAPE]) / param["NAIVE_sMAPE"],
            mean(results_df[:, :MASE]) / param["NAIVE_MASE"],
        ]);
        digits=3,
    )
    crps = round(mean(results_df[:, :CRPS]); digits=3)
    median_crps = round(median(results_df[:, :CRPS]); digits=3)
    summary_df = DataFrame(
        "Model" => ["SS"],
        "MASE" => [mase],
        "sMAPE" => [smape],
        "OWA" => [owa],
        "CRPS" => [crps],
        "Median CRPS" => [median_crps],
    )
    CSV.write("paper_tests/m4_test/metrics_results/BENCHMARK_$(param["name"])_SUMMARY.csv", summary_df)
    return results_df
end

# Main script
function main()
    for gran in keys(parameters_dict)
        results_table = DataFrame()
        for outlier in [true, false]
            for selection in ["split", "fixed_alpha"]
                if selection == "fixed_alpha"
                    information_criteria = "aic"
                    alpha_set = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0] 
                else 
                    information_criteria = "aic"
                    alpha_set = [-1.0]
                end
                for α in alpha_set
                    @info "Running configuration: Param=$(gran), Outlier=$(outlier), IC=$(information_criteria), α=$(α)"
                    results_table = run_config(
                        complete_dict_vec[gran], results_table, outlier, selection, information_criteria, α, parameters_dict[gran]
                    )
                end
            end
        end
        filename = parameters_dict[gran]["name"]
        CSV.write("paper_tests/m4_test/metrics_results/SSL_$(filename)_METRICS_RESULTS.csv", results_table)
        @info "Running benchmark model for $(parameters_dict[gran]["name"])"
        run_benchmark_model(complete_dict_vec[gran], parameters_dict[gran])
    end
end

function create_dirs()
    try
        mkdir("paper_tests/m4_test/results_SSL")
    catch
        @warn "Directory already exists"
    end
    try
        mkdir("paper_tests/m4_test/metrics_results")
    catch
        @warn "Directory already exists"
    end
end

create_dirs()

main()

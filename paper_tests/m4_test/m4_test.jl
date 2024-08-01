import Pkg
Pkg.activate(".")
Pkg.instantiate()

using StateSpaceLearning, CSV, DataFrames, Statistics, Revise

#df_train = CSV.read("paper_tests/m4_test/Monthly-train.csv", DataFrame)
df_train1 = CSV.read("paper_tests/m4_test/Monthly-train1.csv", DataFrame)
df_train2 = CSV.read("paper_tests/m4_test/Monthly-train2.csv", DataFrame)
df_train3 = CSV.read("paper_tests/m4_test/Monthly-train3.csv", DataFrame)
df_train4 = CSV.read("paper_tests/m4_test/Monthly-train4.csv", DataFrame)
df_train = vcat(df_train1, df_train2, df_train3, df_train4) # so that files are not too big and can be uploaded to github
df_test  = CSV.read("paper_tests/m4_test/Monthly-test.csv", DataFrame)

include("metrics.jl")
include("evaluate_model.jl")
include("prepare_data.jl")

dict_vec = build_train_test_dict(df_train, df_test)

H = 18

# Function to append results to CSV file
function append_results(filepath, results_df)
    if isfile(filepath)
        df_old = CSV.read(filepath, DataFrame)
        results_df = vcat(df_old, results_df)
    end
    CSV.write(filepath, results_df)
end

function run_config(results_table::DataFrame, outlier::Bool, information_criteria::String, α::Float64, save_init::Bool, sample_size::Int64)

    NAIVE_sMAPE = 14.427 #M4 Paper
    NAIVE_MASE = 1.063   #M4 Paper

    results_df = DataFrame()
    initialization_df = DataFrame()
    filepath = "paper_tests/m4_test/results_SSL/SSL_$(information_criteria)_$(α)_$(outlier).csv"
    init_filepath = "paper_tests/m4_test/init_SSL/SSL_$(information_criteria)_$(α)_$(outlier).csv"
    !save_init ? CSV.write(filepath, results_df) : nothing# Initialize empty CSV
    save_init ? CSV.write(init_filepath, initialization_df) : nothing # Initialize empty CSV

    for i in 1:48000
        if i in [10001, 20001, 30001, 40001] # Clear DataFrame to save memory
            results_df = DataFrame()
            initialization_df = DataFrame()
        end

        initialization_df, results_df = evaluate_SSL(initialization_df, results_df, dict_vec[i], outlier, α, H, sample_size, information_criteria)

        if i in [10000, 20000, 30000, 40000, 48000]
            !save_init ? append_results(filepath, results_df) : nothing
            save_init ? append_results(init_filepath, initialization_df) : nothing
        end
    end

    results_df = CSV.read(filepath, DataFrame)

    mase  = trunc(mean(results_df[:, :MASE]), digits = 3)
    smape = trunc(mean(results_df[:, :sMAPE]), digits = 3)
    owa   = trunc(mean([mean(results_df[:, :sMAPE])/NAIVE_sMAPE, mean(results_df[:, :MASE])/NAIVE_MASE]), digits = 3)
    name = outlier ? "SSL-O ($(information_criteria), α = $(α))" : "SSL ($(information_criteria), α = $(α))"
    results_table = vcat(results_table, DataFrame("Names" => ["$name"], "MASE" => [mase], "sMAPE" => [smape], "OWA" => [owa]))
    return results_table
end

# Main script
function main()
    results_table = DataFrame()
    for outlier in [true, false]
        for information_criteria in ["aic", "bic"]
            for α in vcat([0.0], collect(0.1:0.2:0.9), [1.0])
                @info "Running SSL with outlier = $outlier, information_criteria = $information_criteria, α = $α"
                results_table = run_config(results_table, outlier, information_criteria, α, false, 60)
            end
        end
    end
    CSV.write("paper_tests/m4_test/metrics_results/SSL_METRICS_RESULTS.csv", results_table)
end 

main()

run_config(DataFrame(), false, "aic", 0.1, true, 2794)#max sample size
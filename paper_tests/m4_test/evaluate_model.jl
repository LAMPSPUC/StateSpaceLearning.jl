function evaluate_SSL(initialization_df::DataFrame, results_df::DataFrame, input::Dict, outlier::Bool, α::Float64, H::Int64, sample_size::Int64, information_criteria::String)
    
    normalized_y = input["normalized_train"]
    y_train      = input["train"]
    y_test       = input["test"]
    max_y        = input["max"]
    min_y        = input["min"]

    T= length(normalized_y)
    normalized_y = normalized_y[max(1, T-sample_size+1):end]
    output = StateSpaceLearning.fit_model(normalized_y; estimation_input=Dict("α" => α, "information_criteria" => information_criteria, "ϵ" => 0.05, "penalize_exogenous" => true, "penalize_initial_states" => true),outlier=outlier,ζ_ω_threshold=12)
    normalized_prediction = StateSpaceLearning.forecast(output, H)
    prediction = de_normalize(normalized_prediction, max_y, min_y)

    mase  = MASE(y_train, y_test, prediction)
    smape = sMAPE(y_test, prediction)

    results_df = vcat(results_df, DataFrame([[mase], [smape]], [:MASE, :sMAPE]))
    initialization_df = vcat(initialization_df, DataFrame([[output.residuals_variances["ξ"]], [output.residuals_variances["ω"]], [output.residuals_variances["ε"]], [output.residuals_variances["ζ"]]], [:ξ, :ω, :ϵ, :ζ]))
    return initialization_df, results_df

end
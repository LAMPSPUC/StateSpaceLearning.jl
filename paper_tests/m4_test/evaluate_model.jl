function evaluate_SSL(
    initialization_df::DataFrame,
    results_df::DataFrame,
    input::Dict,
    outlier::Bool,
    α::AbstractFloat,
    H::Int,
    sample_size::Int,
    information_criteria::String,
)
    normalized_y = input["normalized_train"]
    y_train = input["train"]
    y_test = input["test"]
    max_y = input["max"]
    min_y = input["min"]

    T = length(normalized_y)
    normalized_y = normalized_y[max(1, T - sample_size + 1):end]

    model = StateSpaceLearning.StructuralModel(
        normalized_y;
        level="stochastic",
        slope="stochastic",
        seasonal="stochastic",
        freq_seasonal=12,
        outlier=outlier,
        ζ_threshold=12,
        ω_threshold=12,
    )
    StateSpaceLearning.fit!(
        model;
        α=α,
        information_criteria=information_criteria,
        ϵ=0.05,
        penalize_exogenous=true,
        penalize_initial_states=true,
    )

    normalized_prediction = StateSpaceLearning.forecast(model, H)
    prediction = de_normalize(normalized_prediction, max_y, min_y)

    normalized_scenarios = StateSpaceLearning.simulate(model, H, 1000)
    scenarios = de_normalize(normalized_scenarios, max_y, min_y)

    mase = MASE(y_train, y_test, prediction)
    smape = sMAPE(y_test, prediction)
    crps = CRPS(scenarios, y_test)

    results_df = vcat(
        results_df, DataFrame([[mase], [smape], [crps]], [:MASE, :sMAPE, :CRPS])
    )
    initialization_df = vcat(
        initialization_df,
        DataFrame(
            [
                [model.output.residuals_variances["ξ"]],
                [model.output.residuals_variances["ω_12"]],
                [model.output.residuals_variances["ε"]],
                [model.output.residuals_variances["ζ"]],
            ],
            [:ξ, :ω, :ϵ, :ζ],
        ),
    )
    return initialization_df, results_df
end

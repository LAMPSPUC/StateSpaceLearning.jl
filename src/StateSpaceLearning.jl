module StateSpaceLearning

using LinearAlgebra, Statistics, GLMNet

include("structs.jl")
include("models/autoregressive_locallineartrend.jl")
include("estimation_procedure/information_criteria.jl")
include("estimation_procedure/lasso.jl")
include("estimation_procedure/adalasso.jl")
include("estimation_procedure/estimation_utils.jl")
include("utils.jl")

export fit_model, forecast

"""
    fit_model(y::Vector{Fl}; model_type::String="Basic Structural", Exogenous_X::Union{Matrix{Fl}, Missing}=missing,
              estimation_procedure::String="AdaLasso", s::Int64=12, outlier::Bool=false, stabilize_ζ::Int64=0,
              α::Float64=0.1, hyperparameter_selection::String="aic", ψ::Float64=0.1,
              penalize_exogenous::Bool=true)::Output where Fl

    Fits the StateSpaceLearning model using specified parameters and estimation procedures.

    # Arguments
    - `y::Vector{Fl}`: Vector of data.
    - `model_type::String`: Type of model (default: "Basic Structural").
    - `Exogenous_X::Union{Matrix{Fl}, Missing}`: Exogenous variables matrix (default: missing).
    - `estimation_procedure::String`: Estimation procedure (default: "AdaLasso").
    - `s::Int64`: Seasonal period (default: 12).
    - `outlier::Bool`: Flag for considering outlier component (default: false).
    - `stabilize_ζ::Int64`: Stabilize_ζ parameter (default: 0).
    - `α::Float64`: Elastic net control factor between ridge (α=0) and lasso (α=1) (default: 0.1).
    - `hyperparameter_selection::String`: Information criteria method for hyperparameter selection (default: "aic").
    - `ψ::Float64`: AdaLasso adjustment coefficient (default: 0.05).
    - `penalize_exogenous::Bool`: Flag to select exogenous variables. When false the penalty factor for these variables will be set to 0 (default: true).
    - `penalize_initial_states::Bool`: Flag to penalize initial states. When false the penalty factor for these variables will be set to 0 (default: true).
    - `model_dict::Dict`: Dictionary containing the model functions (default: unobserved_components_dict).
    - `exog_model_args::Dict`: Dictionary containing the exogenous model arguments (default: Dict()).

    # Returns
    - `Output`: Output object containing model information, coefficients, residuals, etc.

"""
function fit_model(y::Vector{Fl}; model_type::String="Basic Structural", Exogenous_X::Union{Matrix{Fl}, Missing}=missing,
                    estimation_procedure::String="AdaLasso", s::Int64=12, outlier::Bool=false, stabilize_ζ::Int64=0,
                    α::Float64=0.1, hyperparameter_selection::String="aic", ψ::Float64=0.05, 
                    penalize_exogenous::Bool=true, penalize_initial_states::Bool=true,
                    model_dict::Dict = unobserved_components_dict, exog_model_args::Dict = Dict())::Output where Fl

    T = length(y)
    @assert T > s "Time series must be longer than the seasonal period"
    @assert 0 <= α <= 1 "α must be in (0, 1], Lasso.jl cannot handle α = 0"

    Exogenous_X = ismissing(Exogenous_X) ? zeros(T, 0) : Exogenous_X

    POSSIBLE_ARGS = Dict("y" => y, "model_type" => model_type, "Exogenous_X" => Exogenous_X,
                         "s" => s, "outlier" => outlier, "stabilize_ζ" => stabilize_ζ, "T" => T)

    for key in keys(exog_model_args) POSSIBLE_ARGS[key] = exog_model_args[key]  end

    X = model_dict["create_X"]([POSSIBLE_ARGS[key] for key in model_dict["create_X_ARGS"]]...)

    invalid_indexes = unique(vcat([i[1] for i in findall(i -> any(isnan, i), X)], findall(i -> isnan(i), y)))
    valid_indexes   = setdiff(1:T, invalid_indexes)
    estimation_y  = y[valid_indexes]
    Estimation_X = X[valid_indexes, :]

    components_indexes  = model_dict["get_components_indexes"]([POSSIBLE_ARGS[key] for key in model_dict["get_components_indexes_ARGS"]]...)

    coefs, estimation_ϵ = fit_estimation_procedure(estimation_procedure, Estimation_X, estimation_y, α, hyperparameter_selection, components_indexes, ψ, penalize_exogenous, penalize_initial_states)

    components          = build_components(Estimation_X, coefs, components_indexes)

    residuals_variances = model_dict["get_variances"](estimation_ϵ, coefs, components_indexes)

    ϵ, fitted = get_fit_and_residuals(estimation_ϵ, coefs, X, valid_indexes, T)

    return Output(model_type, X, coefs, ϵ, fitted, components, residuals_variances, s, T, outlier, valid_indexes, stabilize_ζ, y)
end

"""
    forecast(output::Output, steps_ahead::Int64; Exogenous_Forecast::Union{Matrix{Fl}, Missing}=missing)::Vector{Float64} where Fl

    Returns the forecast for a given number of steps ahead using the provided StateSpaceLearning output and exogenous forecast data.

    # Arguments
    - `output::Output`: Output object obtained from model fitting.
    - `steps_ahead::Int64`: Number of steps ahead for forecasting.
    - `Exogenous_Forecast::Union{Matrix{Fl}, Missing}`: Exogenous variables forecast (default: missing).
    - `model_dict::Dict`: Dictionary containing the model functions (default: unobserved_components_dict).
    - `exog_model_args::Dict`: Dictionary containing the exogenous model arguments (default: Dict()).

    # Returns
    - `Vector{Float64}`: Vector containing forecasted values.

"""
function forecast(output::Output, steps_ahead::Int64; Exogenous_Forecast::Union{Matrix{Fl}, Missing}=missing, model_dict::Dict = unobserved_components_dict, exog_model_args::Dict = Dict())::Vector{Float64} where Fl
    @assert steps_ahead > 0 "steps_ahead must be a positive integer"
    Exogenous_Forecast = ismissing(Exogenous_Forecast) ? zeros(steps_ahead, 0) : Exogenous_Forecast
    
    @assert length(output.components["Exogenous_X"]["Indexes"]) == size(Exogenous_Forecast, 2) "If an exogenous matrix was utilized in the estimation procedure, it must be provided its prediction for the forecast procedure. If no exogenous matrix was utilized, Exogenous_Forecast must be missing"
    @assert size(Exogenous_Forecast, 1) == steps_ahead "Exogenous_Forecast must have the same number of rows as steps_ahead"
    
    POSSIBLE_ARGS = Dict("output" => output, "steps_ahead" => steps_ahead, "Exogenous_Forecast" => Exogenous_Forecast)
    for key in keys(exog_model_args) POSSIBLE_ARGS[key] = exog_model_args[key]  end
    return model_dict["forecast"]([POSSIBLE_ARGS[key] for key in model_dict["forecast_ARGS"]]...)
end

end # module StateSpaceLearning

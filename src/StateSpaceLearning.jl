module StateSpaceLearning

const AVAILABLE_MODELS = ["Basic Structural", "Local Linear Trend", "Local Level"]
const AVAILABLE_ESTIMATION_PROCEDURES = ["Lasso", "AdaLasso"]
const AVAILABLE_HYPERPARAMETER_SELECTION = ["aic", "bic", "aicc", "EBIC"]

using LinearAlgebra, Statistics, GLMNet

include("structs.jl")
include("model_utils.jl")
include("estimation_procedure/information_criteria.jl")
include("estimation_procedure/lasso.jl")
include("estimation_procedure/adalasso.jl")
include("estimation_procedure/estimation_utils.jl")
include("utils.jl")

export fit_model, forecast

"""
    fit_model(y::Vector{Fl}; model_type::String="Basic Structural", Exogenous_X::Union{Matrix{Fl}, Missing}=missing,
              estimation_procedure::String="AdaLasso", s::Int64=12, outlier::Bool=false, stabilize_ζ::Int64=0,
              α::Float64=0.1, hyperparameter_selection::String="aic", adalasso_coef::Float64=0.1,
              select_exogenous::Bool=true)::Output where Fl

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
    - `adalasso_coef::Float64`: AdaLasso adjustment coefficient (default: 0.1).
    - `select_exogenous::Bool`: Flag to select exogenous variables. When false the penalty factor for these variables will be set to 0 (default: true).

    # Returns
    - `Output`: Output object containing model information, coefficients, residuals, etc.

"""
function fit_model(y::Vector{Fl}; model_type::String="Basic Structural", Exogenous_X::Union{Matrix{Fl}, Missing}=missing,
                    estimation_procedure::String="AdaLasso", s::Int64=12, outlier::Bool=false, stabilize_ζ::Int64=0,
                    α::Float64=0.1, hyperparameter_selection::String="aic", adalasso_coef::Float64=0.1, select_exogenous::Bool=true)::Output where Fl

    T = length(y)
    @assert T > s "Time series must be longer than the seasonal period"
    @assert (model_type in AVAILABLE_MODELS) "Unavailable Model"
    @assert (estimation_procedure in AVAILABLE_ESTIMATION_PROCEDURES) "Unavailable estimation procedure"
    @assert (hyperparameter_selection in AVAILABLE_HYPERPARAMETER_SELECTION) "Unavailable hyperparameter selection method"
    @assert 0 <= α <= 1 "α must be in (0, 1], Lasso.jl cannot handle α = 0"
    
    valid_indexes = findall(i -> !isnan(i), y)
    estimation_y  = y[valid_indexes]

    Exogenous_X = ismissing(Exogenous_X) ? zeros(T, 0) : Exogenous_X

    X = create_X(model_type, T, s, Exogenous_X, outlier, stabilize_ζ)
    Estimation_X = X[valid_indexes, :]

    components_indexes  = get_components_indexes(T, s, Exogenous_X, outlier, model_type, stabilize_ζ)

    coefs, estimation_ϵ = fit_estimation_procedure(estimation_procedure, Estimation_X, estimation_y, α, hyperparameter_selection, components_indexes, adalasso_coef, select_exogenous)

    components          = build_components(Estimation_X, coefs, components_indexes)

    residuals_variances = get_variances(estimation_ϵ, coefs, components_indexes)

    ϵ, fitted = build_complete_variables(estimation_ϵ, coefs, X, valid_indexes, T)

    return Output(model_type, X, coefs, ϵ, fitted, components, residuals_variances, s, T, outlier, valid_indexes, stabilize_ζ)
end

"""
    forecast(output::Output, steps_ahead::Int64; Exogenous_Forecast::Union{Matrix{Fl}, Missing}=missing)::Vector{Float64} where Fl

    Returns the forecast for a given number of steps ahead using the provided StateSpaceLearning output and exogenous forecast data.

    # Arguments
    - `output::Output`: Output object obtained from model fitting.
    - `steps_ahead::Int64`: Number of steps ahead for forecasting.
    - `Exogenous_Forecast::Union{Matrix{Fl}, Missing}`: Exogenous variables forecast (default: missing).

    # Returns
    - `Vector{Float64}`: Vector containing forecasted values.

"""
function forecast(output::Output, steps_ahead::Int64; Exogenous_Forecast::Union{Matrix{Fl}, Missing}=missing)::Vector{Float64} where Fl
    @assert steps_ahead > 0 "steps_ahead must be a positive integer"
    Exogenous_Forecast = ismissing(Exogenous_Forecast) ? zeros(steps_ahead, 0) : Exogenous_Forecast
    
    @assert length(output.components["Exogenous_X"]["Indexes"]) == size(Exogenous_Forecast, 2) "If an exogenous matrix was utilized in the estimation procedure, it must be provided its prediction for the forecast procedure. If no exogenous matrix was utilized, Exogenous_Forecast must be missing"
    @assert size(Exogenous_Forecast, 1) == steps_ahead "Exogenous_Forecast must have the same number of rows as steps_ahead"
    
    return forecast_model(output, steps_ahead, Exogenous_Forecast)
end

end # module StateSpaceLearning

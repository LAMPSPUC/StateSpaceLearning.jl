module StateSpaceLearning

using LinearAlgebra, Statistics, GLMNet

include("structs.jl")
include("models/unobserved_components.jl")
include("information_criteria.jl")
include("estimation_procedure/default_estimation_procedure.jl")
include("utils.jl")
include("datasets.jl")

export fit_model, forecast

"""
    fit_model(y::Vector{Fl}; model_type::String="Basic Structural", Exogenous_X::Union{Matrix{Fl}, Missing}=missing,
              estimation_procedure::String="AdaLasso", s::Int64=12, outlier::Bool=false, ζ_ω_threshold::Int64=0,
              α::Float64=0.1, information_criteria::String="aic", ψ::Float64=0.1,
              penalize_exogenous::Bool=true)::Output where Fl

    Fits the StateSpaceLearning model using specified parameters and estimation procedures.

    # Arguments
    - `y::Vector{Fl}`: Vector of data.
    - model_input::Dict: Dictionary containing the model input parameters (default: Dict("level" => true, "stochastic_level" => true, "trend" => true, "stochastic_trend" => true, "seasonal" => true, "stochastic_seasonal" => true, "freq_seasonal" => 12)).
    - estimation_input::Dict: Dictionary containing the estimation input parameters (default: Dict("α" => 0.1, "information_criteria" => "aic", ψ => 0.05, "penalize_exogenous" => true, "penalize_initial_states" => true)).
    - `Exogenous_X::Union{Matrix{Fl}, Missing}`: Exogenous variables matrix (default: missing).
    - `outlier::Bool`: Flag for considering outlier component (default: true).
    - `ζ_ω_threshold::Int64`: ζ_ω_threshold parameter (default: 12).
 

    # Returns
    - `Output`: Output object containing model information, coefficients, residuals, etc.

"""
function fit_model(y::Vector{Fl};
                    model_input::Dict = Dict("stochastic_level" => true, "trend" => true, "stochastic_trend" => true, "seasonal" => true, "stochastic_seasonal" => true, "freq_seasonal" => 12),
                    estimation_input::Dict = Dict("α" => 0.1, "information_criteria" => "aic", "ψ" => 0.05, "penalize_exogenous" => true, "penalize_initial_states" => true),
                    Exogenous_X::Matrix{Fl} = zeros(length(y), 0),
                    outlier::Bool = true, ζ_ω_threshold::Int64 = 12)::Output where Fl

    T = length(y)
    @assert T > model_input["freq_seasonal"] "Time series must be longer than the seasonal period"

    X = create_X_unobserved_components(model_input, Exogenous_X, outlier, ζ_ω_threshold, T)

    estimation_y, Estimation_X, valid_indexes = handle_missing_values(X, y)

    components_indexes  = get_components_indexes(T, Exogenous_X, model_input, outlier, ζ_ω_threshold)

    coefs, estimation_ε = default_estimation_procedure(Estimation_X, estimation_y, components_indexes, estimation_input)

    components          = build_components(Estimation_X, coefs, components_indexes)

    residuals_variances = get_variances(estimation_ε, coefs, components_indexes)

    ε, fitted = get_fit_and_residuals(estimation_ε, coefs, X, valid_indexes, T)

    return Output(model_input, X, coefs, ε, fitted, components, residuals_variances, T, outlier, valid_indexes, ζ_ω_threshold, y)
end

"""
    forecast(output::Output, steps_ahead::Int64; Exogenous_Forecast::Union{Matrix{Fl}, Missing}=missing)::Vector{Float64} where Fl

    Returns the forecast for a given number of steps ahead using the provided StateSpaceLearning output and exogenous forecast data.

    # Arguments
    - `output::Output`: Output object obtained from model fitting.
    - `steps_ahead::Int64`: Number of steps ahead for forecasting.
    - `Exogenous_Forecast::Matrix{Fl}`: Exogenous variables forecast (default: zeros(steps_ahead, 0)).
    - `model_dict::Dict`: Dictionary containing the model functions (default: unobserved_components_dict).
    - `exog_model_args::Dict`: Dictionary containing the exogenous model arguments (default: Dict()).

    # Returns
    - `Vector{Float64}`: Vector containing forecasted values.

"""
function forecast(output::Output, steps_ahead::Int64; Exogenous_Forecast::Matrix{Fl}=zeros(steps_ahead, 0))::Vector{Float64} where Fl
    
    @assert length(output.components["Exogenous_X"]["Indexes"]) == size(Exogenous_Forecast, 2) "If an exogenous matrix was utilized in the estimation procedure, it must be provided its prediction for the forecast procedure. If no exogenous matrix was utilized, Exogenous_Forecast must be missing"
    @assert size(Exogenous_Forecast, 1) == steps_ahead "Exogenous_Forecast must have the same number of rows as steps_ahead"
    
    return forecast_unobserved_components(output, steps_ahead, Exogenous_Forecast)
end

end # module StateSpaceLearning

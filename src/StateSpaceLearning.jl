module StateSpaceLearning

using LinearAlgebra, Statistics, GLMNet, Distributions

include("structs.jl")
include("models/default_model.jl")
include("information_criteria.jl")
include("estimation_procedure/default_estimation_procedure.jl")
include("utils.jl")
include("datasets.jl")

const DEFAULT_COMPONENTS_PARAMETERS = ["level", "stochastic_level", "trend", "stochastic_trend", "seasonal", "stochastic_seasonal", "freq_seasonal"]

export fit_model, forecast

"""
fit_model(y::Vector{Fl};
            model_input::Dict = Dict("level" => true, "stochastic_level" => true, "trend" => true, "stochastic_trend" => true, 
                                        "seasonal" => true, "stochastic_seasonal" => true, "freq_seasonal" => 12,
                                        "outlier" => true, "ζ_ω_threshold" => 12),
            model_functions::Dict = Dict("create_X" => create_X, "get_components_indexes" => get_components_indexes,
                                        "get_variances" => get_variances),
            estimation_input::Dict = Dict("α" => 0.1, "information_criteria" => "aic", "ϵ" => 0.05, 
                                            "penalize_exogenous" => true, "penalize_initial_states" => true),
            estimation_function::Function = default_estimation_procedure,
            Exogenous_X::Matrix{Fl} = zeros(length(y), 0))::Output where Fl

    Fits the StateSpaceLearning model using specified parameters and estimation procedures.

    # Arguments
    - `y::Vector{Fl}`: Vector of data.
    - model_input::Dict: Dictionary containing the model input parameters (default: Dict("level" => true, "stochastic_level" => true, "trend" => true, "stochastic_trend" => true, "seasonal" => true, "stochastic_seasonal" => true, "freq_seasonal" => 12, "outlier" => true, , "ζ_ω_threshold" => 12)).
    - model_functions::Dict: Dictionary containing the model functions (default: Dict("create_X" => create_X, "get_components_indexes" => get_components_indexes, "get_variances" => get_variances)).
    - estimation_input::Dict: Dictionary containing the estimation input parameters (default: Dict("α" => 0.1, "information_criteria" => "aic", ϵ => 0.05, "penalize_exogenous" => true, "penalize_initial_states" => true)).
    - `estimation_function::Function`: Estimation function (default: default_estimation_procedure).
    - `Exogenous_X::Union{Matrix{Fl}, Missing}`: Exogenous variables matrix (default: missing). 

    # Returns
    - `Output`: Output object containing model information, coefficients, residuals, etc.

"""
function fit_model(y::Vector{Fl};
                    model_input::Dict = Dict("level" => true, "stochastic_level" => true, "trend" => true, "stochastic_trend" => true, 
                                             "seasonal" => true, "stochastic_seasonal" => true, "freq_seasonal" => 12,
                                             "outlier" => true, "ζ_ω_threshold" => 12),
                    model_functions::Dict = Dict("create_X" => create_X, "get_components_indexes" => get_components_indexes,
                                             "get_variances" => get_variances),
                    estimation_input::Dict = Dict("α" => 0.1, "information_criteria" => "aic", "ϵ" => 0.05, 
                                                   "penalize_exogenous" => true, "penalize_initial_states" => true),
                    estimation_function::Function = default_estimation_procedure,
                    Exogenous_X::Matrix{Fl} = zeros(length(y), 0))::Output where Fl

    T = length(y)
    
    if model_functions["create_X"] == create_X
        @assert T > model_input["freq_seasonal"] "Time series must be longer than the seasonal period"
        @assert all([key in keys(model_input) for key in DEFAULT_COMPONENTS_PARAMETERS]) "The default components model must have all the necessary parameters $(DEFAULT_COMPONENTS_PARAMETERS)"
    end

    @assert !has_intercept(Exogenous_X) "Exogenous matrix must not have an intercept column"

    X = model_functions["create_X"](model_input, Exogenous_X)

    if has_intercept(X)
        @assert allequal(X[:, 1]) "Intercept column must be the first column"
        @assert !has_intercept(X[:, 2:end]) "Matrix must not have more than one intercept column"
    end

    estimation_y, Estimation_X, valid_indexes = handle_missing_values(X, y)

    components_indexes  = model_functions["get_components_indexes"](Exogenous_X, model_input)

    coefs, estimation_ε = estimation_function(Estimation_X, estimation_y, components_indexes, estimation_input)

    components          = build_components(Estimation_X, coefs, components_indexes)

    residuals_variances = model_functions["get_variances"](estimation_ε, coefs, components_indexes)

    ε, fitted           = get_fit_and_residuals(estimation_ε, coefs, X, valid_indexes, T)

    return Output(model_input, model_functions["create_X"], X, coefs, ε, fitted, components, residuals_variances, valid_indexes)
end

"""
    forecast(output::Output, steps_ahead::Int64; Exogenous_Forecast::Union{Matrix{Fl}, Missing}=missing)::Vector{Float64} where Fl

    Returns the forecast for a given number of steps ahead using the provided StateSpaceLearning output and exogenous forecast data.

    # Arguments
    - `output::Output`: Output object obtained from model fitting.
    - `steps_ahead::Int64`: Number of steps ahead for forecasting.
    - `Exogenous_Forecast::Matrix{Fl}`: Exogenous variables forecast (default: zeros(steps_ahead, 0))

    # Returns
    - `Vector{Float64}`: Vector containing forecasted values.

"""
function forecast(output::Output, steps_ahead::Int64; Exogenous_Forecast::Matrix{Fl}=zeros(steps_ahead, 0))::Vector{Float64} where Fl
    
    @assert length(output.components["Exogenous_X"]["Indexes"]) == size(Exogenous_Forecast, 2) "If an exogenous matrix was utilized in the estimation procedure, it must be provided its prediction for the forecast procedure. If no exogenous matrix was utilized, Exogenous_Forecast must be missing"
    @assert size(Exogenous_Forecast, 1) == steps_ahead "Exogenous_Forecast must have the same number of rows as steps_ahead"
    
    Exogenous_X = output.X[:, output.components["Exogenous_X"]["Indexes"]]
    complete_matrix = output.Create_X(output.model_input, Exogenous_X, steps_ahead, Exogenous_Forecast)

    return complete_matrix[end-steps_ahead+1:end, :]*output.coefs
end

"""
simulate(output::Output, steps_ahead::Int64; N_scenarios::Int64 = 1000, simulate_outliers::Bool = true, Exogenous_Forecast::Matrix{Fl}=zeros(steps_ahead, 0))::Matrix{Float64} where Fl

    Generate simulations for a given number of steps ahead using the provided StateSpaceLearning output and exogenous forecast data.

    # Arguments
    - `output::Output`: Output object obtained from model fitting.
    - `steps_ahead::Int64`: Number of steps ahead for simulation.
    - `N_scenarios::Int64`: Number of scenarios to simulate (default: 1000).
    - `simulate_outliers::Bool`: If true, simulate outliers (default: true).
    - `Exogenous_Forecast::Matrix{Fl}`: Exogenous variables forecast (default: zeros(steps_ahead, 0))

    # Returns
    - `Matrix{Float64}`: Matrix containing simulated values.
"""
function simulate(output::Output, steps_ahead::Int64, N_scenarios::Int64; simulate_outliers::Bool = true, 
                  innovation_functions::Dict = Dict("stochastic_level" => Dict("create_X" => create_ξ, "component" => "ξ", "args" => (length(output.ε) + steps_ahead + 1, 0)),
                                                "stochastic_trend" => Dict("create_X" => create_ζ, "component" => "ζ", "args" => (length(output.ε) + steps_ahead + 1, 0, 1)),
                                                "stochastic_seasonal" => Dict("create_X" => create_ω, "component" => "ω", "args" => (length(output.ε) + steps_ahead + 1, output.model_input["freq_seasonal"], 0, 1))),
                  Exogenous_Forecast::Matrix{Fl}=zeros(steps_ahead, 0))::Matrix{Float64} where Fl

    prediction = forecast(output, steps_ahead; Exogenous_Forecast = Exogenous_Forecast)

    T = length(output.ε)
    simulation_X = zeros(steps_ahead, 0)
    components_matrix = zeros(length(output.valid_indexes), 0)
    N_components = 1

    for innovation in keys(innovation_functions)
        if output.model_input[innovation]
            innov_dict = innovation_functions[innovation]
            simulation_X = hcat(simulation_X, innov_dict["create_X"](innov_dict["args"]...)[end-steps_ahead:end-1, end-steps_ahead+1:end])
            comp = fill_innovation_coefs(T, innov_dict["component"], output)
            components_matrix = hcat(components_matrix, comp[output.valid_indexes])
            N_components += 1
        end
    end
   
    components_matrix = hcat(components_matrix, output.ε[output.valid_indexes])
    simulation_X = hcat(simulation_X, Matrix(1.0 * I, steps_ahead, steps_ahead))
    components_matrix += rand(Normal(0, 1), size(components_matrix)) ./ 1e9 # Make sure matrix is positive definite

    ∑ = cov(components_matrix)
    MV_dist = MvNormal(zeros(N_components), ∑)
    o_noises = simulate_outliers && output.model_input["outlier"] ? rand(Normal(0, std(output.components["o"]["Coefs"])), steps_ahead, N_scenarios) : zeros(steps_ahead, N_scenarios)

    simulation = hcat([prediction for _ in 1:N_scenarios]...)
    for s in 1:N_scenarios
        sim_coefs = ones(size(simulation_X, 2)) .* NaN
        
        for i in 1:steps_ahead
            rand_inovs = rand(MV_dist)
        
            for comp in eachindex(rand_inovs)
                sim_coefs[i + (comp - 1) * steps_ahead] = rand_inovs[comp]
            end
        end
        
        simulation[:, s] += (simulation_X * sim_coefs + o_noises[:, s])
    end

    return simulation

end

end # module StateSpaceLearning


"""
function fit!(model::StateSpaceLearningModel,
                    α::Float64 = 0.1, 
                    information_criteria::String = "aic", 
                    ϵ::Float64 = 0.05, 
                    penalize_exogenous::Bool = true, 
                    penalize_initial_states::Bool = true,
                    )

    Fits the StateSpaceLearning model using specified parameters and estimation procedures.

    # Arguments
    model::StateSpaceLearningModel: Model to be fitted.
    α::Float64: Elastic net mixing parameter (default: 0.1).
    information_criteria::String: Method for hyperparameter selection (default: "aic").
    ϵ::Float64: Non negative value to handle 0 coefs on the first lasso step (default: 0.05).
    penalize_exogenous::Bool: If true, penalize exogenous variables (default: true).
    penalize_initial_states::Bool: If true, penalize initial states (default: true).
"""
function fit!(
    model::StateSpaceLearningModel;
    α::Float64=0.1,
    information_criteria::String="aic",
    ϵ::Float64=0.05,
    penalize_exogenous::Bool=true,
    penalize_initial_states::Bool=true,
)
    if has_intercept(model.X)
        @assert allequal(model.X[:, 1]) "Intercept column must be the first column"
        @assert !has_intercept(model.X[:, 2:end]) "Matrix must not have more than one intercept column"
    end

    estimation_y, Estimation_X, valid_indexes = handle_missing_values(model.X, model.y)

    components_indexes = get_components_indexes(model)

    coefs, estimation_ε = estimation_procedure(
        Estimation_X,
        estimation_y,
        components_indexes,
        α,
        information_criteria,
        ϵ,
        penalize_exogenous,
        penalize_initial_states,
    )

    components = build_components(Estimation_X, coefs, components_indexes)

    residuals_variances = get_variances(model, estimation_ε, coefs, components_indexes)

    ε, fitted = get_fit_and_residuals(
        estimation_ε, coefs, model.X, valid_indexes, length(model.y)
    )

    output = Output(coefs, ε, fitted, residuals_variances, valid_indexes, components)
    return model.output = output
end

"""
    forecast(model::StateSpaceLearningModel, steps_ahead::Int; Exogenous_Forecast::Union{Matrix{Fl}, Missing}=missing)::Vector{Float64} where Fl

    Returns the forecast for a given number of steps ahead using the provided StateSpaceLearning output and exogenous forecast data.

    # Arguments
    - `model::StateSpaceLearningModel`: Model obtained from fitting.
    - `steps_ahead::Int`: Number of steps ahead for forecasting.
    - `Exogenous_Forecast::Matrix{Fl}`: Exogenous variables forecast (default: zeros(steps_ahead, 0))

    # Returns
    - `Vector{Float64}`: Vector containing forecasted values.

"""
function forecast(
    model::StateSpaceLearningModel,
    steps_ahead::Int;
    Exogenous_Forecast::Matrix{Fl}=zeros(steps_ahead, 0),
)::Vector{Float64} where {Fl}
    @assert length(model.output.components["Exogenous_X"]["Indexes"]) ==
        size(Exogenous_Forecast, 2) "If an exogenous matrix was utilized in the estimation procedure, it must be provided its prediction for the forecast procedure. If no exogenous matrix was utilized, Exogenous_Forecast must be missing"
    @assert size(Exogenous_Forecast, 1) == steps_ahead "Exogenous_Forecast must have the same number of rows as steps_ahead"

    Exogenous_X = model.X[:, model.output.components["Exogenous_X"]["Indexes"]]
    complete_matrix = create_X(
        model.level,
        model.stochastic_level,
        model.trend,
        model.stochastic_trend,
        model.seasonal,
        model.stochastic_seasonal,
        model.freq_seasonal,
        model.outlier,
        model.ζ_ω_threshold,
        Exogenous_X,
        steps_ahead,
        Exogenous_Forecast,
    )

    return complete_matrix[(end - steps_ahead + 1):end, :] * model.output.coefs
end

"""
simulate(model::StateSpaceLearningModel, steps_ahead::Int, N_scenarios::Int;
                                 Exogenous_Forecast::Matrix{Fl}=zeros(steps_ahead, 0))::Matrix{Float64} where Fl

    Generate simulations for a given number of steps ahead using the provided StateSpaceLearning output and exogenous forecast data.

    # Arguments
    - `model::StateSpaceLearningModel`: Model obtained from fitting.
    - `steps_ahead::Int`: Number of steps ahead for simulation.
    - `N_scenarios::Int`: Number of scenarios to simulate (default: 1000).
    - `Exogenous_Forecast::Matrix{Fl}`: Exogenous variables forecast (default: zeros(steps_ahead, 0))

    # Returns
    - `Matrix{Float64}`: Matrix containing simulated values.
"""
function simulate(
    model::StateSpaceLearningModel,
    steps_ahead::Int,
    N_scenarios::Int;
    Exogenous_Forecast::Matrix{Fl}=zeros(steps_ahead, 0),
)::Matrix{Float64} where {Fl}
    prediction = forecast(model, steps_ahead; Exogenous_Forecast=Exogenous_Forecast)

    simulation_X = zeros(steps_ahead, 0)
    components_matrix = zeros(length(model.output.valid_indexes), 0)
    N_components = 1

    model_innovations = get_model_innovations(model)
    for innovation in model_innovations
        if innovation in keys(model.output.residuals_variances)
            simulation_X = hcat(
                simulation_X,
                get_innovation_simulation_X(model, innovation, steps_ahead)[
                    (end - steps_ahead):(end - 1), (end - steps_ahead + 1):end
                ],
            )
            comp = fill_innovation_coefs(model, innovation)
            components_matrix = hcat(components_matrix, comp[model.output.valid_indexes])
            N_components += 1
        end
    end

    components_matrix = hcat(components_matrix, model.output.ε[model.output.valid_indexes])
    simulation_X = hcat(simulation_X, Matrix(1.0 * I, steps_ahead, steps_ahead))
    components_matrix += rand(Normal(0, 1), size(components_matrix)) ./ 1e9 # Make sure matrix is positive definite

    ∑ = cov(components_matrix)
    MV_dist = MvNormal(zeros(N_components), ∑)
    o_noises = if model.outlier
        rand(Normal(0, std(model.output.components["o"]["Coefs"])), steps_ahead, N_scenarios)
    else
        zeros(steps_ahead, N_scenarios)
    end

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

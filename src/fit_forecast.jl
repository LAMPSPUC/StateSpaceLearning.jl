
@doc raw"""
Fits the StateSpaceLearning model using specified parameters and estimation procedures. After fitting, the model output can be accessed in model.output

fit!(model::StateSpaceLearningModel,
                    α::AbstractFloat = 0.1, 
                    information\_criteria::String = "aic", 
                    ϵ::AbstractFloat = 0.05, 
                    penalize\_exogenous::Bool = true, 
                    penalize\_initial\_states::Bool = true,
                    )

# Arguments
- model::StateSpaceLearningModel: Model to be fitted.
- α::AbstractFloat: Elastic net mixing parameter (default: 0.1).
- information\_criteria::String: Method for hyperparameter selection (default: "aic").
- ϵ::AbstractFloat: Non negative value to handle 0 coefs on the first lasso step (default: 0.05).
- penalize\_exogenous::Bool: If true, penalize exogenous variables (default: true).
- penalize\_initial\_states::Bool: If true, penalize initial states (default: true).

# Example
```julia
y = rand(100)
model = StructuralModel(y)
fit!(model)
output = model.output
```

"""
function fit!(
    model::StateSpaceLearningModel;
    α::AbstractFloat=0.1,
    information_criteria::String="aic",
    ϵ::AbstractFloat=0.05,
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

    T = typeof(model.y) <: Vector ? length(model.y) : size(model.y, 1)

    ε, fitted = get_fit_and_residuals(estimation_ε, coefs, model.X, valid_indexes, T)

    components_ts = get_components_ts(model, components)

    if typeof(model.y) <: Vector
        output = Output(
            coefs, ε, fitted, residuals_variances, valid_indexes, components, components_ts
        )
    else
        output = Output[]
        for i in eachindex(coefs)
            push!(
                output,
                Output(
                    coefs[i],
                    ε[i],
                    fitted[i],
                    residuals_variances[i],
                    valid_indexes,
                    components[i],
                    components_ts[i],
                ),
            )
        end
    end
    return model.output = output
end

@doc raw"""
Returns the forecast for a given number of steps ahead using the provided StateSpaceLearning output and exogenous forecast data.

forecast(model::StateSpaceLearningModel, steps\_ahead::Int; Exogenous\_Forecast::Union{Matrix{Fl}, Missing}=missing)::Vector{AbstractFloat} where Fl

# Arguments
- `model::StateSpaceLearningModel`: Model obtained from fitting.
- `steps_ahead::Int`: Number of steps ahead for forecasting.
- `Exogenous_Forecast::Matrix{Fl}`: Exogenous variables forecast (default: zeros(steps_ahead, 0))

# Returns
- `Union{Matrix{AbstractFloat}, Vector{AbstractFloat}}`: Matrix or vector of matrices containing forecasted values.

# Example
```julia
y = rand(100)
model = StructuralModel(y)
fit!(model)
steps_ahead = 12
point_prediction = forecast(model, steps_ahead)
```
"""
function forecast(
    model::StateSpaceLearningModel,
    steps_ahead::Int;
    Exogenous_Forecast::Matrix{Fl}=zeros(steps_ahead, 0),
)::Union{Matrix{<:AbstractFloat},Vector{<:AbstractFloat}} where {Fl<:AbstractFloat}
    @assert isfitted(model) "Model must be fitted before simulation"
    exog_idx = if typeof(model.output) == Output
        model.output.components["Exogenous_X"]["Indexes"]
    else
        model.output[1].components["Exogenous_X"]["Indexes"]
    end
    @assert length(exog_idx) == size(Exogenous_Forecast, 2) "If an exogenous matrix was utilized in the estimation procedure, it must be provided its prediction for the forecast procedure. If no exogenous matrix was utilized, Exogenous_Forecast must be missing"
    @assert size(Exogenous_Forecast, 1) == steps_ahead "Exogenous_Forecast must have the same number of rows as steps_ahead"

    Exogenous_X = model.X[:, exog_idx]
    complete_matrix = create_X(model, Exogenous_X, steps_ahead, Exogenous_Forecast)

    if typeof(model.output) == Output
        return AbstractFloat.(
            complete_matrix[(end - steps_ahead + 1):end, :] * model.output.coefs
        )
    else
        prediction = Matrix{AbstractFloat}(undef, steps_ahead, length(model.output))
        for i in eachindex(model.output)
            prediction[:, i] =
                complete_matrix[(end - steps_ahead + 1):end, :] * model.output[i].coefs
        end
        return AbstractFloat.(prediction)
    end
end

@doc raw"""
Generate simulations for a given number of steps ahead using the provided StateSpaceLearning output and exogenous forecast data.

simulate(model::StateSpaceLearningModel, steps\_ahead::Int, N\_scenarios::Int;
                                 Exogenous\_Forecast::Matrix{Fl}=zeros(steps_ahead, 0))::Matrix{AbstractFloat} where Fl

# Arguments
- `model::StateSpaceLearningModel`: Model obtained from fitting.
- `steps_ahead::Int`: Number of steps ahead for simulation.
- `N_scenarios::Int`: Number of scenarios to simulate (default: 1000).
- `Exogenous_Forecast::Matrix{Fl}`: Exogenous variables forecast (default: zeros(steps_ahead, 0))

# Returns
- `Union{Vector{Matrix{AbstractFloat}}, Matrix{AbstractFloat}}`: Matrix or vector of matrices containing simulated values.

# Example (Univariate Case)
```julia
y = rand(100)
model = StructuralModel(y)
fit!(model)
steps_ahead = 12
N_scenarios = 1000
simulation  = simulate(model, steps_ahead, N_scenarios)
```

# Example (Multivariate Case)
```julia
y = rand(100, 3)
model = StructuralModel(y)
fit!(model)
steps_ahead = 12
N_scenarios = 1000
simulation  = simulate(model, steps_ahead, N_scenarios)
```
"""
function simulate(
    model::StateSpaceLearningModel,
    steps_ahead::Int,
    N_scenarios::Int;
    Exogenous_Forecast::Matrix{Fl}=zeros(steps_ahead, 0),
    seasonal_innovation_simulation::Int=0,
)::Union{Vector{Matrix{<:AbstractFloat}},Matrix{<:AbstractFloat}} where {Fl<:AbstractFloat}
    @assert seasonal_innovation_simulation >= 0 "seasonal_innovation_simulation must be a non-negative integer"
    @assert seasonal_innovation_simulation >= 0 "seasonal_innovation_simulation must be a non-negative integer"
    @assert isfitted(model) "Model must be fitted before simulation"

    prediction = forecast(model, steps_ahead; Exogenous_Forecast=Exogenous_Forecast)

    is_univariate = typeof(model.output) == Output

    simulation_X = zeros(steps_ahead, 0)
    valid_indexes =
        is_univariate ? model.output.valid_indexes : model.output[1].valid_indexes
    components_matrix = zeros(length(valid_indexes), 0)
    N_components = 1

    model_innovations = get_model_innovations(model)
    for innovation in model_innovations
        simulation_X = hcat(
            simulation_X,
            get_innovation_simulation_X(model, innovation, steps_ahead)[
                (end - steps_ahead):(end - 1), (end - steps_ahead + 1):end
            ],
        )
        comp = fill_innovation_coefs(model, innovation, valid_indexes)
        components_matrix = hcat(components_matrix, comp)
        N_components += 1
    end

    if is_univariate
        components_matrix = hcat(components_matrix, model.output.ε[valid_indexes])
        @assert N_components < length(model.y)//seasonal_innovation_simulation "The parameter `seasonal_innovation_simulation` is too large for the given dataset, please reduce it"
    else
        for i in eachindex(model.output)
            components_matrix = hcat(components_matrix, model.output[i].ε[valid_indexes])
        end
        N_mv_components = N_components * length(model.output)
        @assert N_mv_components < size(model.y, 1)//seasonal_innovation_simulation "The parameter `seasonal_innovation_simulation` is too large for the given dataset, please reduce it"
    end
    simulation_X = hcat(simulation_X, Matrix(1.0 * I, steps_ahead, steps_ahead))
    components_matrix += rand(Normal(0, 1), size(components_matrix)) ./ 1e9 # Make sure matrix is positive definite

    MV_dist_vec = Vector{MvNormal}(undef, steps_ahead)
    o_noises = if is_univariate
        zeros(steps_ahead, N_scenarios)
    else
        [zeros(steps_ahead, N_scenarios) for _ in 1:length(model.output)]
    end

    if seasonal_innovation_simulation == 0
        ∑ = if is_univariate
            Diagonal([var(components_matrix[:, i]) for i in 1:N_components])
        else
            Diagonal([var(components_matrix[:, i]) for i in 1:N_mv_components])
        end
        for i in 1:steps_ahead
            MV_dist_vec[i] = if is_univariate
                MvNormal(zeros(N_components), ∑)
            else
                MvNormal(zeros(N_mv_components), ∑)
            end
        end

        if model.outlier
            if is_univariate
                o_noises = rand(
                    Normal(0, std(model.output.components["o"]["Coefs"])),
                    steps_ahead,
                    N_scenarios,
                )
            else
                o_noises = [
                    rand(
                        Normal(0, std(model.output[i].components["o"]["Coefs"])),
                        steps_ahead,
                        N_scenarios,
                    ) for i in eachindex(model.output)
                ]
            end
        end
    else
        start_seasonal_term = (size(components_matrix, 1) % seasonal_innovation_simulation)
        for i in 1:min(seasonal_innovation_simulation, steps_ahead)
            ∑ = if is_univariate
                Diagonal([
                    var(
                        components_matrix[
                            (i + start_seasonal_term):seasonal_innovation_simulation:end,
                            j,
                        ],
                    ) for j in 1:N_components
                ])
            else
                Diagonal([
                    var(
                        components_matrix[
                            (i + start_seasonal_term):seasonal_innovation_simulation:end,
                            j,
                        ],
                    ) for j in 1:N_mv_components
                ])
            end

            MV_dist_vec[i] = if is_univariate
                MvNormal(zeros(N_components), ∑)
            else
                MvNormal(zeros(N_mv_components), ∑)
            end
            if is_univariate
                if model.outlier
                    o_noises[i, :] = rand(
                        Normal(
                            0,
                            std(
                                model.output.components["o"]["Coefs"][(i + start_seasonal_term):seasonal_innovation_simulation:end],
                            ),
                        ),
                        N_scenarios,
                    )
                else
                    nothing
                end
            else
                for j in eachindex(model.output)
                    if model.outlier
                        o_noises[j][i, :] = rand(
                            Normal(
                                0,
                                std(
                                    model.output[j].components["o"]["Coefs"][(i + start_seasonal_term):seasonal_innovation_simulation:end],
                                ),
                            ),
                            N_scenarios,
                        )
                    else
                        nothing
                    end
                end
            end
        end
        for i in (seasonal_innovation_simulation + 1):steps_ahead
            MV_dist_vec[i] = MV_dist_vec[i - seasonal_innovation_simulation]
            if model.outlier
                if is_univariate
                    o_noises[i, :] = o_noises[i - seasonal_innovation_simulation, :]
                else
                    for j in eachindex(model.output)
                        o_noises[j][i, :] = o_noises[j][
                            i - seasonal_innovation_simulation, :,
                        ]
                    end
                end
            end
        end
    end

    simulation = if is_univariate
        AbstractFloat.(hcat([prediction for _ in 1:N_scenarios]...))
    else
        [
            AbstractFloat.(hcat([prediction[:, i] for _ in 1:N_scenarios]...)) for
            i in eachindex(model.output)
        ]
    end
    if is_univariate
        fill_simulation!(simulation, MV_dist_vec, o_noises, simulation_X)
    else
        fill_simulation!(
            simulation, MV_dist_vec, o_noises, simulation_X, length(model_innovations)
        )
        simulation = Vector{Matrix{<:AbstractFloat}}(simulation)
    end

    return simulation
end

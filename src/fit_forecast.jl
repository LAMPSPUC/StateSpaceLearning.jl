
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

    components_indexes_orig = get_components_indexes(model)

    innovations_names = get_model_innovations(model)

    estimation_y, Estimation_X, valid_indexes, valid_columns, components_indexes = handle_missing_values(model.X, model.y, components_indexes_orig, innovations_names)

    coefs_raw, estimation_ε = estimation_procedure(
        Estimation_X,
        estimation_y,
        components_indexes,
        α,
        information_criteria,
        ϵ,
        penalize_exogenous,
        penalize_initial_states,
    )

    coefs = zeros(size(model.X, 2))
    coefs[valid_columns] = coefs_raw

    components = build_components(model.X, coefs, components_indexes_orig)

    residuals_variances = get_variances(model, estimation_ε, coefs, components_indexes_orig)

    T = typeof(model.y) <: Vector ? length(model.y) : size(model.y, 1)

    ε, fitted = get_fit_and_residuals(estimation_ε, coefs, model.X, valid_indexes, T)

    decomposition = get_model_decomposition(model, components)

    return model.output = Output(
        coefs, ε, fitted, residuals_variances, valid_indexes, components, decomposition
    )
end

@doc raw"""
Fits the StateSpaceLearning model using a train/validation split and selects α from a set by minimizing validation error.

fit_split!(
    model::StateSpaceLearningModel;
    H::Int=1,
    α_set::Vector{Fl}=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    information_criteria::String="aic",
    ϵ::AbstractFloat=0.05,
    penalize_exogenous::Bool=true,
    penalize_initial_states::Bool=true,
    seed::Int=123
)

# Arguments
- model::StateSpaceLearningModel: Model to be fitted.
- H::Int: Validation horizon (default: 1).
- α_set::Vector{<:AbstractFloat}: Candidate α values for selection (default: [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]).
- information\_criteria::String: Method for hyperparameter selection passed to fit! (default: "aic").
- ϵ::AbstractFloat: Non negative value to handle 0 coefs on the first lasso step (default: 0.05).
- penalize\_exogenous::Bool: If true, penalize exogenous variables (default: true).
- penalize\_initial\_states::Bool: If true, penalize initial states (default: true).
- seed::Int: Random seed for reproducibility (default: 123).

# Example
```julia
y = rand(100)
model = StructuralModel(y)
fit_split!(model; H=10)
output = model.output
```

"""
function fit_split!(
    model::StateSpaceLearningModel;
    H::Int=1,
    α_set::Vector{Fl}=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    information_criteria::String="aic",
    ϵ::AbstractFloat=0.05,
    penalize_exogenous::Bool=true,
    penalize_initial_states::Bool=true,
    seed::Int=123
) where {Fl<:AbstractFloat}

    # model parameters
    level = (model.level && model.stochastic_level) ? "stochastic" : model.level ? "deterministic" : "none"
    slope = (model.slope && model.stochastic_slope) ? "stochastic" : model.slope ? "deterministic" : "none"
    seasonal = (model.seasonal && model.stochastic_seasonal) ? "stochastic" : model.seasonal ? "deterministic" : "none"
    cycle = (model.cycle && model.stochastic_cycle) ? "stochastic" : model.cycle ? "deterministic" : "none"
    @assert model.n_exogenous == 0 "Exogenous variables are not supported in this method yet"
    @assert isnothing(model.dynamic_exog_coefs) "Dynamic exogenous coefficients are not supported in this method yet"
    ##############################################################################

    Random.seed!(seed)

    train_idx = collect(1:length(model.y) - H)
    val_idx = collect(length(model.y) - H + 1:length(model.y))

    train_y = model.y[train_idx]
    val_y = model.y[val_idx]

    rmse_vec = AbstractFloat[]
    for α in α_set
        model_α = StructuralModel(train_y; level=level, slope=slope, seasonal=seasonal, cycle=cycle, freq_seasonal=model.freq_seasonal, cycle_period=model.cycle_period, outlier=model.outlier, ξ_threshold=model.ξ_threshold, ζ_threshold=model.ζ_threshold, ω_threshold=model.ω_threshold, ϕ_threshold=model.ϕ_threshold, stochastic_start=model.stochastic_start)
        fit!(model_α; α=α, information_criteria=information_criteria, ϵ=ϵ, penalize_exogenous=penalize_exogenous,penalize_initial_states=penalize_initial_states)
        prediction = forecast(model_α, H)
        push!(rmse_vec, mean((val_y - prediction) .^ 2))
    end

    fit!(model; α=α_set[argmin(rmse_vec)], information_criteria=information_criteria, ϵ=ϵ, penalize_exogenous=penalize_exogenous,penalize_initial_states=penalize_initial_states)
end
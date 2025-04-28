
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

    innovations_names = get_model_innovations(model)

    coefs, estimation_ε = estimation_procedure(
        Estimation_X,
        estimation_y,
        components_indexes,
        α,
        information_criteria,
        ϵ,
        penalize_exogenous,
        penalize_initial_states,
        innovations_names,
    )

    components = build_components(Estimation_X, coefs, components_indexes)

    residuals_variances = get_variances(model, estimation_ε, coefs, components_indexes)

    T = typeof(model.y) <: Vector ? length(model.y) : size(model.y, 1)

    ε, fitted = get_fit_and_residuals(estimation_ε, coefs, model.X, valid_indexes, T)

    decomposition = get_model_decomposition(model, components)

    return model.output = Output(
        coefs, ε, fitted, residuals_variances, valid_indexes, components, decomposition
    )
end

# Add New Models

The StateSpaceLearning framework supports any additive state space formulation. This section illustrates how to utilize the framework for a specific model.

## Local Level Model

Although the Local Level Model is already implemented within the scope of unobserved components, we use it here as an example. To incorporate a new model, it is necessary to create a dictionary containing the model inputs and another dictionary containing three functions (create_X, get_components_indexes, and get_variances).

### Model Inputs
For the Local Level Model, no parameters are needed. Thus, the model input can be created as:

```julia
model_input = Dict()
```

### create_X
The create_X function constructs the matrices in the State Space Learning format. It must accept the following inputs: (model_input::Dict, Exogenous_X::Matrix{Fl}, outlier::Bool, ζ_ω_threshold::Int64, T::Int64, steps_ahead::Int64=0, Exogenous_Forecast::Matrix{Fl}=zeros(steps_ahead, size(Exogenous_X, 2))). This function may not use parameters such as outlier, ζ_ω_threshold, or Exogenous_X. It must return a matrix.

```julia
function create_X_LocalLevel(model_input::Dict, Exogenous_X::Matrix{Fl}, outlier::Bool, ζ_ω_threshold::Int64, T::Int64,
                  steps_ahead::Int64=0, Exogenous_Forecast::Matrix{Fl}=zeros(steps_ahead, size(Exogenous_X, 2))) where Fl
    initial_states_matrix = ones(T+steps_ahead, 1)
    ξ_matrix = Matrix{Float64}(undef, T+steps_ahead, T - 1)
    for t in 1:T+steps_ahead
        ξ_matrix[t, :] = t < T ? vcat(ones(t-1), zeros(T-t)) : ones(T-1)
    end
    
    return hcat(initial_states_matrix, ξ_matrix)
end
```

### get_components_indexes
The get_components_indexes function outputs a dictionary indicating the indexes of each model component, including a set of indexes for all initial states. For the Local Level Model, the only components are the initial state μ1 and its innovations ξ. The function must accept the following inputs: (T::Int64, Exogenous_X::Matrix{Fl}, model_input::Dict, outlier::Bool, ζ_ω_threshold::Int64). This function may not use parameters such as outlier, ζ_ω_threshold, or Exogenous_X. It must return a dictionary.

```julia
function get_components_indexes_LocalLevel(T::Int64, Exogenous_X::Matrix{Fl}, model_input::Dict, outlier::Bool, ζ_ω_threshold::Int64)::Dict where Fl
    μ1_indexes = [1]
    initial_states_indexes = [1]
    ξ_indexes = collect(2:T)
    return Dict("μ1" => μ1_indexes, "ξ" => ξ_indexes, "initial_states" => initial_states_indexes)
end
```
### get_variances
The get_variances function calculates the variances of the innovations and residuals. It must accept the following inputs:(ε::Vector{Fl}, coefs::Vector{Fl}, components_indexes::Dict{String, Vector{Int64}}). It must return a dictionary.

```julia
function get_variances_LocalLevel(ε::Vector{Fl}, coefs::Vector{Fl}, components_indexes::Dict{String, Vector{Int64}})::Dict where Fl
    
    variances = Dict()
    variances["ξ"] = var(coefs[components_indexes["ξ"]])
    variances["ε"] = var(ε)
    return variances
end
```

### Running the new model
To test the new model, run the fit_model function with the new inputs:

```julia
using StateSpaceLearning
using Statistics

y = randn(100)

#Fit Model
output = StateSpaceLearning.fit_model(y; model_input = model_input, model_functions = Dict("create_X" => create_X_LocalLevel, 
                                       "get_components_indexes" =>   get_components_indexes_LocalLevel, "get_variances" => get_variances_LocalLevel))
```

# Changing Estimation Procedure
The current estimation procedure is based on an Adaptive Lasso. However, alternative methods can be chosen within the StateSpaceLearning framework. Below is an example of how to implement a simple model that minimizes the sum of squares of the residuals. This requires creating two variables: a dictionary estimation_input (which is empty in this case) and a function estimation_function with the following arguments:(Estimation_X::Matrix{Tl}, estimation_y::Vector{Fl}, components_indexes::Dict{String, Vector{Int64}}, estimation_input::Dict). The function should return a tuple containing the model coefficients and residuals.

```julia
estimation_input = Dict()
function estimation_function_min_sq(Estimation_X::Matrix{Tl}, estimation_y::Vector{Fl}, components_indexes::Dict{String, Vector{Int64}}, estimation_input::Dict) where {Tl, Fl}
    mq_coefs = Estimation_X \ estimation_y
    mq_res = estimation_y - (Estimation_X * mq_coefs)
    return mq_coefs, mq_res
end
```
### Running the model with the new estimation procedure
```julia
using StateSpaceLearning

y = randn(100)

#Fit Model
output = StateSpaceLearning.fit_model(y; estimation_input = estimation_input, estimation_function = estimation_function_min_sq)
```

By following these steps, you can customize and extend the StateSpaceLearning framework to suit a variety of state space models and estimation procedures.
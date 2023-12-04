"""
    fit_adalasso(Estimation_X::Matrix{Tl}, estimation_y::Vector{Fl}, α::Float64,
                 hyperparameter_selection::String,
                 components_indexes::Dict{String, Vector{Int64}},
                 adalasso_coef::Float64, select_exogenous::Bool)::Tuple{Vector{Float64}, Vector{Float64}} where {Tl, Fl}

    Fits an Adaptive Lasso (AdaLasso) regression model to the provided data and returns coefficients and residuals.

    # Arguments
    - `Estimation_X::Matrix{Tl}`: Matrix of predictors for estimation.
    - `estimation_y::Vector{Fl}`: Vector of response values for estimation.
    - `α::Float64`: Elastic net control factor between ridge (α=0) and lasso (α=1) (default: 0.1).
    - `hyperparameter_selection::String`: Information Criteria method for hyperparameter selection (default: aic).
    - `components_indexes::Dict{String, Vector{Int64}}`: Dictionary containing indexes for different components.
    - `adalasso_coef::Float64`: AdaLasso adjustment coefficient (default: 0.1).
    - `select_exogenous::Bool`: Flag for selecting exogenous variables. When false the penalty factor for these variables will be set to 0.
    - `penalize_initial_states::Bool`: Flag to penalize initial states. When false the penalty factor for these variables will be set to 0.

    # Returns
    - `Tuple{Vector{Float64}, Vector{Float64}}`: Tuple containing coefficients and residuals of the fitted AdaLasso model.

"""
function fit_adalasso(Estimation_X::Matrix{Tl}, estimation_y::Vector{Fl}, α::Float64, 
                        hyperparameter_selection::String, 
                        components_indexes::Dict{String, Vector{Int64}},
                        adalasso_coef::Float64, select_exogenous::Bool, penalize_initial_states::Bool)::Tuple{Vector{Float64}, Vector{Float64}} where {Tl, Fl}

    penalty_factor = ones(size(Estimation_X, 2) - 1); penalty_factor[components_indexes["initial_states"][2:end] .- 1] .= 0
    coefs, _  = fit_lasso(Estimation_X, estimation_y, α, hyperparameter_selection, select_exogenous, components_indexes; penalty_factor = penalty_factor, intercept = false)

    #AdaLasso per component
    penalty_factor = zeros(size(Estimation_X, 2) - 1)
    for key in keys(components_indexes)
        if key != "initial_states" && key != "μ₁"
            component = components_indexes[key]
            if key != "Exogenous_X" && key != "o" && !(key in ["ν₁", "γ₁"])
                κ = count(i -> i != 0, coefs[component]) < 1 ? 0 : std(coefs[component])
                penalty_factor[component .- 1] .= (1 / (κ + adalasso_coef))
            else
                penalty_factor[component .- 1]  = (1 ./ (abs.(coefs[component]) .+ adalasso_coef))
            end
        end
    end 
    !penalize_initial_states ? penalty_factor[components_indexes["initial_states"][2:end] .- 1] .= 0 : nothing
    return fit_lasso(Estimation_X, estimation_y, α, hyperparameter_selection, select_exogenous, components_indexes; penalty_factor=penalty_factor)
end


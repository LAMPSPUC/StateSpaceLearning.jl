"""
    get_dummy_indexes(Exogenous_X::Matrix{Fl}) where {Fl}

    Identifies and returns the indexes of columns in the exogenous matrix that contain dummy variables.

    # Arguments
    - `Exogenous_X::Matrix{Fl}`: Exogenous variables matrix.

    # Returns
    - `Vector{Int}`: Vector containing the indexes of columns with dummy variables.

"""
function get_dummy_indexes(Exogenous_X::Matrix{Fl}) where{Fl}
    
    T, p = size(Exogenous_X)
    dummy_indexes = []

    for i in 1:p
        if count(iszero.(Exogenous_X[:, i])) == T - 1
            push!(dummy_indexes, findfirst(i -> i != 0.0, Exogenous_X[:, i]))
        end
    end

    return dummy_indexes
end

"""
    get_outlier_duplicate_columns(Estimation_X::Matrix{Tl}, components_indexes::Dict{String, Vector{Int64}}) where{Tl}

    Identifies and returns the indexes of outlier columns that are duplicates of dummy variables in the exogenous matrix.

    # Arguments
    - `Estimation_X::Matrix{Tl}`: Matrix used for estimation.
    - `components_indexes::Dict{String, Vector{Int64}}`: Dictionary containing indexes for different components.

    # Returns
    - `Vector{Int}`: Vector containing the indexes of outlier columns that are duplicates of dummy variables in the exogenous matrix.

"""
function get_outlier_duplicate_columns(Estimation_X::Matrix{Tl}, components_indexes::Dict{String, Vector{Int64}}) where{Tl}
    o_indexes = components_indexes["o"]
    exogenous_indexes = components_indexes["Exogenous_X"]

    dummy_indexes = get_dummy_indexes(Estimation_X[:, exogenous_indexes])

    return o_indexes[dummy_indexes] .- 1
end

"""
    get_path_information_criteria(model::GLMNetPath, Estimation_X::Matrix{Tl}, estimation_y::Vector{Fl},
        information_criteria::String; intercept::Bool = true)::Tuple{Vector{Float64}, Vector{Float64}} where {Tl, Fl}

    Calculates the information criteria along the regularization path of a GLMNet model and returns coefficients and residuals of the best model based on the selected information criteria.

    # Arguments
    - `model::GLMNetPath`: Fitted GLMNetPath model object.
    - `Estimation_X::Matrix{Tl}`: Matrix of predictors for estimation.
    - `estimation_y::Vector{Fl}`: Vector of response values for estimation.
    - `information_criteria::String`: Information Criteria method for hyperparameter selection.
    - `intercept::Bool`: Flag for intercept inclusion in the model (default: true).

    # Returns
    - `Tuple{Vector{Float64}, Vector{Float64}}`: Tuple containing coefficients and residuals of the best model.

"""
function get_path_information_criteria(model::GLMNetPath, Estimation_X::Matrix{Tl}, estimation_y::Vector{Fl}, information_criteria::String; intercept::Bool = true)::Tuple{Vector{Float64}, Vector{Float64}} where {Tl, Fl}
    path_size = length(model.lambda)
    T         = size(Estimation_X, 1)
    K         = count(i->i != 0, model.betas; dims = 1)'

    method_vec = Vector{Float64}(undef, path_size)
    for i in 1:path_size
        fit = Estimation_X*model.betas[:, i] .+ model.a0[i]
        ϵ   = estimation_y - fit
        
        method_vec[i] = get_information(T, K[i], ϵ; information_criteria = information_criteria)
    end

    best_model_idx = argmin(method_vec)
    coefs = intercept ? vcat(model.a0[best_model_idx], model.betas[:, best_model_idx]) : model.betas[:, best_model_idx]
    fit   = intercept ? hcat(ones(T), Estimation_X)*coefs : Estimation_X*coefs
    ϵ   = estimation_y - fit
    return coefs, ϵ
end

"""
    fit_glmnet(Estimation_X::Matrix{Tl}, estimation_y::Vector{Fl}, α::Float64;
               information_criteria::String = "aic",
               penalty_factor::Vector{Float64}=ones(size(Estimation_X,2) - 1),
               intercept::Bool = intercept)::Tuple{Vector{Float64}, Vector{Float64}} where {Tl, Fl}

    Fits a GLMNet model to the provided data and returns coefficients and residuals based on selected criteria.

    # Arguments
    - `Estimation_X::Matrix{Tl}`: Matrix of predictors for estimation.
    - `estimation_y::Vector{Fl}`: Vector of response values for estimation.
    - `α::Float64`: Elastic net control factor between ridge (α=0) and lasso (α=1) (default: 0.1).
    - `information_criteria::String`: Information Criteria method for hyperparameter selection (default: aic).
    - `penalty_factor::Vector{Float64}`: Penalty factors for each predictor (default: ones(size(Estimation_X, 2) - 1)).
    - `intercept::Bool`: Flag for intercept inclusion in the model (default: true).

    # Returns
    - `Tuple{Vector{Float64}, Vector{Float64}}`: Tuple containing coefficients and residuals of the best model.

"""
function fit_glmnet(Estimation_X::Matrix{Tl}, estimation_y::Vector{Fl}, α::Float64; information_criteria::String = "aic", penalty_factor::Vector{Float64}=ones(size(Estimation_X,2) - 1), intercept::Bool = intercept)::Tuple{Vector{Float64}, Vector{Float64}} where {Tl, Fl}
    model = glmnet(Estimation_X, estimation_y, alpha = α, penalty_factor = penalty_factor, intercept = intercept, dfmax=size(Estimation_X, 2), lambda_min_ratio=0.001)
    return get_path_information_criteria(model, Estimation_X, estimation_y, information_criteria; intercept = intercept)
end

"""
    fit_lasso(Estimation_X::Matrix{Tl}, estimation_y::Vector{Fl}, α::Float64, information_criteria::String,
              penalize_exogenous::Bool, components_indexes::Dict{String, Vector{Int64}};
              penalty_factor::Vector{Float64}=ones(size(Estimation_X,2) - 1), intercept::Bool = true)::Tuple{Vector{Float64}, Vector{Float64}} where {Tl, Fl}

    Fits a Lasso regression model to the provided data and returns coefficients and residuals based on selected criteria.

    # Arguments
    - `Estimation_X::Matrix{Tl}`: Matrix of predictors for estimation.
    - `estimation_y::Vector{Fl}`: Vector of response values for estimation.
    - `α::Float64`: Elastic net control factor between ridge (α=0) and lasso (α=1) (default: 0.1).
    - `information_criteria::String`: Information Criteria method for hyperparameter selection (default: aic).
    - `penalize_exogenous::Bool`: Flag for selecting exogenous variables. When false the penalty factor for these variables will be set to 0.
    - `components_indexes::Dict{String, Vector{Int64}}`: Dictionary containing indexes for different components.
    - `penalty_factor::Vector{Float64}`: Penalty factors for each predictor (default: ones(size(Estimation_X, 2) - 1)).
    - `intercept::Bool`: Flag for intercept inclusion in the model (default: true).

    # Returns
    - `Tuple{Vector{Float64}, Vector{Float64}}`: Tuple containing coefficients and residuals of the fitted Lasso model.

"""
function fit_lasso(Estimation_X::Matrix{Tl}, estimation_y::Vector{Fl}, α::Float64, information_criteria::String, penalize_exogenous::Bool, components_indexes::Dict{String, Vector{Int64}}; penalty_factor::Vector{Float64}=ones(size(Estimation_X,2) - 1), intercept::Bool = true)::Tuple{Vector{Float64}, Vector{Float64}} where {Tl, Fl}

    outlier_duplicate_columns = get_outlier_duplicate_columns(Estimation_X, components_indexes)
    penalty_factor[outlier_duplicate_columns] .= Inf

    !penalize_exogenous ? penalty_factor[components_indexes["Exogenous_X"] .- 1] .= 0 : nothing
    mean_y = mean(estimation_y); Lasso_y = intercept ? estimation_y : estimation_y .- mean_y

    coefs, ϵ =  fit_glmnet(Estimation_X[:, 2:end], Lasso_y, α; information_criteria=information_criteria, penalty_factor=penalty_factor, intercept = intercept)
    return !intercept ? (vcat(mean_y, coefs), ϵ) : (coefs, ϵ)
    
end

"""
    fit_adalasso(Estimation_X::Matrix{Tl}, estimation_y::Vector{Fl}, α::Float64,
                 information_criteria::String,
                 components_indexes::Dict{String, Vector{Int64}},
                 ψ::Float64, penalize_exogenous::Bool)::Tuple{Vector{Float64}, Vector{Float64}} where {Tl, Fl}

    Fits an Adaptive Lasso (AdaLasso) regression model to the provided data and returns coefficients and residuals.

    # Arguments
    - `Estimation_X::Matrix{Tl}`: Matrix of predictors for estimation.
    - `estimation_y::Vector{Fl}`: Vector of response values for estimation.
    - `components_indexes::Dict{String, Vector{Int64}}`: Dictionary containing indexes for different components.
    - `estimation_input::Dict`: Dictionary containing the estimation input parameters.

    # Returns
    - `Tuple{Vector{Float64}, Vector{Float64}}`: Tuple containing coefficients and residuals of the fitted AdaLasso model.

"""
function default_estimation_procedure(Estimation_X::Matrix{Tl}, estimation_y::Vector{Fl}, 
                        components_indexes::Dict{String, Vector{Int64}}, estimation_input::Dict)::Tuple{Vector{Float64}, Vector{Float64}} where {Tl, Fl}

    α = estimation_input["α"]; information_criteria = estimation_input["information_criteria"]; 
    ψ = estimation_input["ψ"]; penalize_exogenous = estimation_input["penalize_exogenous"]; 
    penalize_initial_states = estimation_input["penalize_initial_states"]

    @assert 0 <= α <= 1 "α must be in [0, 1]"

    penalty_factor = ones(size(Estimation_X, 2) - 1); penalty_factor[components_indexes["initial_states"][2:end] .- 1] .= 0
    coefs, _  = fit_lasso(Estimation_X, estimation_y, α, information_criteria, penalize_exogenous, components_indexes; penalty_factor = penalty_factor, intercept = false)

    #AdaLasso per component
    penalty_factor = zeros(size(Estimation_X, 2) - 1)
    for key in keys(components_indexes)
        if key != "initial_states" && key != "μ1"
            component = components_indexes[key]
            if key != "Exogenous_X" && key != "o" && !(key in ["ν1", "γ1"])
                κ = count(i -> i != 0, coefs[component]) < 1 ? 0 : std(coefs[component])
                penalty_factor[component .- 1] .= (1 / (κ + ψ))
            else
                penalty_factor[component .- 1]  = (1 ./ (abs.(coefs[component]) .+ ψ))
            end
        end
    end 
    !penalize_initial_states ? penalty_factor[components_indexes["initial_states"][2:end] .- 1] .= 0 : nothing
    return fit_lasso(Estimation_X, estimation_y, α, information_criteria, penalize_exogenous, components_indexes; penalty_factor=penalty_factor)
end

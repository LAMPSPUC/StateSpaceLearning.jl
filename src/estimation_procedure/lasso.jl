"""
    get_path_information_criteria(model::GLMNetPath, Estimation_X::Matrix{Tl}, estimation_y::Vector{Fl},
        hyperparameter_selection::String; intercept::Bool = true)::Tuple{Vector{Float64}, Vector{Float64}} where {Tl, Fl}

    Calculates the information criteria along the regularization path of a GLMNet model and returns coefficients and residuals of the best model based on the selected information criteria.

    # Arguments
    - `model::GLMNetPath`: Fitted GLMNetPath model object.
    - `Estimation_X::Matrix{Tl}`: Matrix of predictors for estimation.
    - `estimation_y::Vector{Fl}`: Vector of response values for estimation.
    - `hyperparameter_selection::String`: Information Criteria method for hyperparameter selection.
    - `intercept::Bool`: Flag for intercept inclusion in the model (default: true).

    # Returns
    - `Tuple{Vector{Float64}, Vector{Float64}}`: Tuple containing coefficients and residuals of the best model.

"""
function get_path_information_criteria(model::GLMNetPath, Estimation_X::Matrix{Tl}, estimation_y::Vector{Fl}, hyperparameter_selection::String; intercept::Bool = true)::Tuple{Vector{Float64}, Vector{Float64}} where {Tl, Fl}
    path_size = length(model.lambda)
    T, p      = size(Estimation_X)
    K         = count(i->i != 0, model.betas; dims = 1)'

    method_vec = Vector{Float64}(undef, path_size)
    for i in 1:path_size
        fit = Estimation_X*model.betas[:, i] .+ model.a0[i]
        ϵ   = estimation_y - fit
        
        method_vec[i] = get_information(T, K[i], ϵ; hyperparameter_selection = hyperparameter_selection, p = p)
    end

    best_model_idx = argmin(method_vec)
    coefs = intercept ? vcat(model.a0[best_model_idx], model.betas[:, best_model_idx]) : model.betas[:, best_model_idx]
    fit   = intercept ? hcat(ones(T), Estimation_X)*coefs : Estimation_X*coefs
    ϵ   = estimation_y - fit
    return coefs, ϵ
end

"""
    fit_glmnet(Estimation_X::Matrix{Tl}, estimation_y::Vector{Fl}, α::Float64;
               hyperparameter_selection::String = "aic",
               penalty_factor::Vector{Float64}=ones(size(Estimation_X,2) - 1),
               intercept::Bool = intercept)::Tuple{Vector{Float64}, Vector{Float64}} where {Tl, Fl}

    Fits a GLMNet model to the provided data and returns coefficients and residuals based on selected criteria.

    # Arguments
    - `Estimation_X::Matrix{Tl}`: Matrix of predictors for estimation.
    - `estimation_y::Vector{Fl}`: Vector of response values for estimation.
    - `α::Float64`: Elastic net control factor between ridge (α=0) and lasso (α=1) (default: 0.1).
    - `hyperparameter_selection::String`: Information Criteria method for hyperparameter selection (default: aic).
    - `penalty_factor::Vector{Float64}`: Penalty factors for each predictor (default: ones(size(Estimation_X, 2) - 1)).
    - `intercept::Bool`: Flag for intercept inclusion in the model (default: true).

    # Returns
    - `Tuple{Vector{Float64}, Vector{Float64}}`: Tuple containing coefficients and residuals of the best model.

"""
function fit_glmnet(Estimation_X::Matrix{Tl}, estimation_y::Vector{Fl}, α::Float64; hyperparameter_selection::String = "aic", penalty_factor::Vector{Float64}=ones(size(Estimation_X,2) - 1), intercept::Bool = intercept)::Tuple{Vector{Float64}, Vector{Float64}} where {Tl, Fl}
    model = glmnet(Estimation_X, estimation_y, alpha = α, penalty_factor = penalty_factor, intercept = intercept, dfmax=size(Estimation_X, 2), lambda_min_ratio=0.001)
    return get_path_information_criteria(model, Estimation_X, estimation_y, hyperparameter_selection; intercept = intercept)
end

"""
    fit_lasso(Estimation_X::Matrix{Tl}, estimation_y::Vector{Fl}, α::Float64, hyperparameter_selection::String,
              penalize_exogenous::Bool, components_indexes::Dict{String, Vector{Int64}};
              penalty_factor::Vector{Float64}=ones(size(Estimation_X,2) - 1), intercept::Bool = true)::Tuple{Vector{Float64}, Vector{Float64}} where {Tl, Fl}

    Fits a Lasso regression model to the provided data and returns coefficients and residuals based on selected criteria.

    # Arguments
    - `Estimation_X::Matrix{Tl}`: Matrix of predictors for estimation.
    - `estimation_y::Vector{Fl}`: Vector of response values for estimation.
    - `α::Float64`: Elastic net control factor between ridge (α=0) and lasso (α=1) (default: 0.1).
    - `hyperparameter_selection::String`: Information Criteria method for hyperparameter selection (default: aic).
    - `penalize_exogenous::Bool`: Flag for selecting exogenous variables. When false the penalty factor for these variables will be set to 0.
    - `components_indexes::Dict{String, Vector{Int64}}`: Dictionary containing indexes for different components.
    - `penalty_factor::Vector{Float64}`: Penalty factors for each predictor (default: ones(size(Estimation_X, 2) - 1)).
    - `intercept::Bool`: Flag for intercept inclusion in the model (default: true).

    # Returns
    - `Tuple{Vector{Float64}, Vector{Float64}}`: Tuple containing coefficients and residuals of the fitted Lasso model.

"""
function fit_lasso(Estimation_X::Matrix{Tl}, estimation_y::Vector{Fl}, α::Float64, hyperparameter_selection::String, penalize_exogenous::Bool, components_indexes::Dict{String, Vector{Int64}}; penalty_factor::Vector{Float64}=ones(size(Estimation_X,2) - 1), intercept::Bool = true)::Tuple{Vector{Float64}, Vector{Float64}} where {Tl, Fl}

    outlier_duplicate_columns = get_outlier_duplicate_columns(Estimation_X, components_indexes)
    penalty_factor[outlier_duplicate_columns] .= Inf

    !penalize_exogenous ? penalty_factor[components_indexes["Exogenous_X"] .- 1] .= 0 : nothing
    mean_y = mean(estimation_y); Lasso_y = intercept ? estimation_y : estimation_y .- mean_y

    coefs, ϵ =  fit_glmnet(Estimation_X[:, 2:end], Lasso_y, α; hyperparameter_selection=hyperparameter_selection, penalty_factor=penalty_factor, intercept = intercept)
    return !intercept ? (vcat(mean_y, coefs), ϵ) : (coefs, ϵ)
    
end
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

function fit_glmnet(Estimation_X::Matrix{Tl}, estimation_y::Vector{Fl}, α::Float64; hyperparameter_selection::String = "aic", penalty_factor::Vector{Float64}=ones(size(Estimation_X,2) - 1), intercept::Bool = intercept)::Tuple{Vector{Float64}, Vector{Float64}} where {Tl, Fl}
    model = glmnet(Estimation_X, estimation_y, alpha = α, penalty_factor = penalty_factor, intercept = intercept, dfmax=size(Estimation_X, 2), lambda_min_ratio=0.001)
    return get_path_information_criteria(model, Estimation_X, estimation_y, hyperparameter_selection; intercept = intercept)
end

function fit_lasso(Estimation_X::Matrix{Tl}, estimation_y::Vector{Fl}, α::Float64, hyperparameter_selection::String, select_exogenous::Bool, components_indexes::Dict{String, Vector{Int64}}; penalty_factor::Vector{Float64}=ones(size(Estimation_X,2) - 1), intercept::Bool = true)::Tuple{Vector{Float64}, Vector{Float64}} where {Tl, Fl}

    outlier_duplicate_columns = get_outlier_duplicate_columns(Estimation_X, components_indexes)
    penalty_factor[outlier_duplicate_columns] .= Inf

    !select_exogenous ? penalty_factor[components_indexes["Exogenous_X"] .- 1] .= 0 : nothing
    mean_y = mean(estimation_y); Lasso_y = intercept ? estimation_y : estimation_y .- mean_y

    coefs, ϵ =  fit_glmnet(Estimation_X[:, 2:end], Lasso_y, α; hyperparameter_selection=hyperparameter_selection, penalty_factor=penalty_factor, intercept = intercept)
    return !intercept ? (vcat(mean_y, coefs), ϵ) : (coefs, ϵ)
    
end
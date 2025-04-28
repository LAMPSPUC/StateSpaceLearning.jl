"""
    get_dummy_indexes(exog::Matrix{Fl}) where {Fl}

    Identifies and returns the indexes of columns in the exogenous matrix that contain dummy variables.

    # Arguments
    - `exog::Matrix{Fl}`: Exogenous variables matrix.

    # Returns
    - `Vector{Int}`: Vector containing the indexes of columns with dummy variables.

"""
function get_dummy_indexes(exog::Matrix{Fl}) where {Fl}
    T, p = size(exog)
    dummy_indexes = []

    for i in 1:p
        if count(iszero.(exog[:, i])) == T - 1
            push!(dummy_indexes, findfirst(i -> i != 0.0, exog[:, i]))
        end
    end

    return dummy_indexes
end

"""
    get_outlier_duplicate_columns(Estimation_X::Matrix{Fl}, components_indexes::Dict{String, Vector{Int}}) where{Fl}

    Identifies and returns the indexes of outlier columns that are duplicates of dummy variables in the exogenous matrix.

    # Arguments
    - `Estimation_X::Matrix{Fl}`: Matrix used for estimation.
    - `components_indexes::Dict{String, Vector{Int}}`: Dictionary containing indexes for different components.

    # Returns
    - `Vector{Int}`: Vector containing the indexes of outlier columns that are duplicates of dummy variables in the exogenous matrix.

"""
function get_outlier_duplicate_columns(
    Estimation_X::Matrix{Fl}, components_indexes::Dict{String,Vector{Int}}
) where {Fl}
    if !haskey(components_indexes, "o")
        return []
    else
        o_indexes = components_indexes["o"]
        exogenous_indexes = components_indexes["exog"]

        dummy_indexes = get_dummy_indexes(Estimation_X[:, exogenous_indexes])

        return o_indexes[dummy_indexes] .- 1
    end
end

"""
    get_path_information_criteria(
    model::GLMNetPath,
    Lasso_X::Matrix{Tl},
    Lasso_y::Vector{Fl},
    information_criteria::String;
    intercept::Bool=true,
)::Tuple{Vector{AbstractFloat},Vector{AbstractFloat}} where {Fl <: AbstractFloat, Tl <: AbstractFloat}

    Calculates the information criteria along the regularization path of a GLMNet model and returns coefficients and residuals of the best model based on the selected information criteria.

    # Arguments
    - `model::GLMNetPath`: Fitted GLMNetPath model object.
    - `Lasso_X::Matrix{Tl}`: Matrix of predictors for estimation.
    - `Lasso_y::Vector{Fl}`: Vector of response values for estimation.
    - `information_criteria::String`: Information Criteria method for hyperparameter selection.
    - `intercept::Bool`: Flag for intercept inclusion in the model (default: true).

    # Returns
    - `Tuple{Vector{AbstractFloat}, Vector{AbstractFloat}}`: Tuple containing coefficients and residuals of the best model.

"""
function get_path_information_criteria(
    model::GLMNetPath,
    Lasso_X::Matrix{Tl},
    Lasso_y::Vector{Fl},
    information_criteria::String;
    intercept::Bool=true,
)::Tuple{
    Vector{AbstractFloat},Vector{AbstractFloat}
} where {Fl<:AbstractFloat,Tl<:AbstractFloat}
    path_size = length(model.lambda)
    T = size(Lasso_X, 1)
    K = count(i -> i != 0, model.betas; dims=1)'

    method_vec = Vector{AbstractFloat}(undef, path_size)
    for i in 1:path_size
        fit = Lasso_X * model.betas[:, i] .+ model.a0[i]
        ε = Lasso_y - fit

        method_vec[i] = get_information(
            T, K[i], ε; information_criteria=information_criteria
        )
    end

    best_model_idx = argmin(method_vec)
    coefs = if intercept
        vcat(model.a0[best_model_idx], model.betas[:, best_model_idx])
    else
        model.betas[:, best_model_idx]
    end
    fit = intercept ? hcat(ones(T), Lasso_X) * coefs : Lasso_X * coefs
    ε = Lasso_y - fit
    return coefs, ε
end

"""
    fit_glmnet(
    Lasso_X::Matrix{Tl},
    Lasso_y::Vector{Fl},
    α::AbstractFloat;
    information_criteria::String="aic",
    penalty_factor::Vector{Pl}=ones(size(Lasso_X, 2) - 1),
    intercept::Bool=intercept,
)::Tuple{Vector{AbstractFloat},Vector{AbstractFloat}} where {Fl <: AbstractFloat, Tl <: AbstractFloat, Pl <: AbstractFloat}

    Fits a GLMNet model to the provided data and returns coefficients and residuals based on selected criteria.

    # Arguments
    - `Lasso_X::Matrix{Tl}`: Matrix of predictors for estimation.
    - `Lasso_y::Vector{Fl}`: Vector of response values for estimation.
    - `α::AbstractFloat`: Elastic net control factor between ridge (α=0) and lasso (α=1) (default: 0.1).
    - `information_criteria::String`: Information Criteria method for hyperparameter selection (default: aic).
    - `penalty_factor::Vector{Pl}`: Penalty factors for each predictor (default: ones(size(Lasso_X, 2) - 1)).
    - `intercept::Bool`: Flag for intercept inclusion in the model (default: true).

    # Returns
    - `Tuple{Vector{AbstractFloat}, Vector{AbstractFloat}}`: Tuple containing coefficients and residuals of the best model.

"""
function fit_glmnet(
    Lasso_X::Matrix{Tl},
    Lasso_y::Vector{Fl},
    α::AbstractFloat;
    information_criteria::String="aic",
    penalty_factor::Vector{Pl}=ones(size(Lasso_X, 2) - 1),
    intercept::Bool=intercept,
)::Tuple{
    Vector{AbstractFloat},Vector{AbstractFloat}
} where {Fl<:AbstractFloat,Tl<:AbstractFloat,Pl<:AbstractFloat}
    model = glmnet(
        Lasso_X,
        Lasso_y;
        alpha=α,
        penalty_factor=penalty_factor,
        intercept=intercept,
        dfmax=size(Lasso_X, 2),
        lambda_min_ratio=0.001,
    )
    return get_path_information_criteria(
        model, Lasso_X, Lasso_y, information_criteria; intercept=intercept
    )
end

"""
    fit_lasso(
    Estimation_X::Matrix{Tl},
    estimation_y::Vector{Fl},
    α::AbstractFloat,
    information_criteria::String,
    penalize_exogenous::Bool,
    components_indexes::Dict{String,Vector{Int}},
    penalty_factor::Vector{Pl};
    rm_average::Bool=false,
)::Tuple{Vector{AbstractFloat},Vector{AbstractFloat}} where {Fl <: AbstractFloat, Tl <: AbstractFloat, Pl <: AbstractFloat}

    Fits a Lasso regression model to the provided data and returns coefficients and residuals based on selected criteria.

    # Arguments
    - `Estimation_X::Matrix{Fl}`: Matrix of predictors for estimation.
    - `estimation_y::Vector{Fl}`: Vector of response values for estimation.
    - `α::AbstractFloat`: Elastic net control factor between ridge (α=0) and lasso (α=1) (default: 0.1).
    - `information_criteria::String`: Information Criteria method for hyperparameter selection (default: aic).
    - `penalize_exogenous::Bool`: Flag for selecting exogenous variables. When false the penalty factor for these variables will be set to 0.
    - `components_indexes::Dict{String, Vector{Int}}`: Dictionary containing indexes for different components.
    - `penalty_factor::Vector{Fl}`: Penalty factors for each predictor.
    - `rm_average::Bool`: Flag to consider if the intercept will be calculated is the average of the time series (default: false).

    # Returns
    - `Tuple{Vector{AbstractFloat}, Vector{AbstractFloat}}`: Tuple containing coefficients and residuals of the fitted Lasso model.

"""
function fit_lasso(
    Estimation_X::Matrix{Tl},
    estimation_y::Vector{Fl},
    α::AbstractFloat,
    information_criteria::String,
    penalize_exogenous::Bool,
    components_indexes::Dict{String,Vector{Int}},
    penalty_factor::Vector{Pl};
    rm_average::Bool=false,
)::Tuple{
    Vector{AbstractFloat},Vector{AbstractFloat}
} where {Fl<:AbstractFloat,Tl<:AbstractFloat,Pl<:AbstractFloat}
    outlier_duplicate_columns = get_outlier_duplicate_columns(
        Estimation_X, components_indexes
    )
    penalty_factor[outlier_duplicate_columns] .= Inf

    hasintercept = has_intercept(Estimation_X)
    if hasintercept
        if !penalize_exogenous
            penalty_factor[components_indexes["exog"] .- 1] .= 0
        else
            nothing
        end
        Lasso_X = Estimation_X[:, 2:end]
    else
        if !penalize_exogenous
            penalty_factor[components_indexes["exog"]] .= 0
        else
            nothing
        end
        Lasso_X = Estimation_X
        @assert !rm_average "Intercept must be included in the model if rm_average is set to true"
    end

    if rm_average
        mean_y = mean(estimation_y)
        Lasso_y = estimation_y .- mean_y
    else
        Lasso_y = estimation_y
    end

    if hasintercept
        coefs, ε = fit_glmnet(
            Lasso_X,
            Lasso_y,
            α;
            information_criteria=information_criteria,
            penalty_factor=penalty_factor,
            intercept=!rm_average,
        )
    else
        coefs, ε = fit_glmnet(
            Lasso_X,
            Lasso_y,
            α;
            information_criteria=information_criteria,
            penalty_factor=penalty_factor,
            intercept=false,
        )
    end
    return rm_average ? (vcat(mean_y, coefs), ε) : (coefs, ε)
end

"""
    estimation_procedure(
    Estimation_X::Matrix{Tl},
    estimation_y::Vector{Fl},
    components_indexes::Dict{String,Vector{Int}},
    α::AbstractFloat,
    information_criteria::String,
    ϵ::AbstractFloat,
    penalize_exogenous::Bool,
    penalize_initial_states::Bool,
    innovations_names::Vector{String},
    initial_states_names::Vector{String},
)::Tuple{Vector{AbstractFloat},Vector{AbstractFloat}} where {Fl <: AbstractFloat, Tl <: AbstractFloat}

    Fits an Adaptive Lasso (AdaLasso) regression model to the provided data and returns coefficients and residuals.

    # Arguments
    - `Estimation_X::Matrix{Fl}`: Matrix of predictors for estimation.
    - `estimation_y::Vector{Fl}`: Vector of response values for estimation.
    - `components_indexes::Dict{String, Vector{Int}}`: Dictionary containing indexes for different components.
    - `α::AbstractFloat`: Elastic net control factor between ridge (α=0) and lasso (α=1) (default: 0.1).
    - `information_criteria::String`: Information Criteria method for hyperparameter selection (default: aic).
    - `ϵ::AbstractFloat`: Non negative value to handle 0 coefs on the first lasso step (default: 0.05).
    - `penalize_exogenous::Bool`: Flag for selecting exogenous variables. When false the penalty factor for these variables will be set to 0.
    - `penalize_initial_states::Bool`: Flag for selecting initial states. When false the penalty factor for these variables will be set to 0.
    - `innovations_names::Vector{String}`: Vector of strings containing the names of the innovations.
    - `initial_states_names::Vector{String}`: Vector of strings containing the names of the initial states.


    # Returns
    - `Tuple{Vector{AbstractFloat}, Vector{AbstractFloat}}`: Tuple containing coefficients and residuals of the fitted AdaLasso model.

"""
function estimation_procedure(
    Estimation_X::Matrix{Tl},
    estimation_y::Vector{Fl},
    components_indexes::Dict{String,Vector{Int}},
    α::AbstractFloat,
    information_criteria::String,
    ϵ::AbstractFloat,
    penalize_exogenous::Bool,
    penalize_initial_states::Bool,
    innovations_names::Vector{String},
)::Tuple{
    Vector{AbstractFloat},Vector{AbstractFloat}
} where {Fl<:AbstractFloat,Tl<:AbstractFloat}
    @assert 0 <= α <= 1 "α must be in [0, 1]"
    @assert ϵ > 0 "ϵ must be positive"

    hasintercept = has_intercept(Estimation_X)

    # all zero columns in X 
    all_zero_idx = findall(i -> all(iszero, Estimation_X[:, i]), 1:size(Estimation_X, 2))

    if hasintercept
        penalty_factor = ones(size(Estimation_X, 2) - 1)
        if length(penalty_factor) != length(components_indexes["initial_states"][2:end])
            penalty_factor[components_indexes["initial_states"][2:end] .- 1] .= 0
        end
        penalty_factor[all_zero_idx .- 1] .= Inf
        coefs, _ = fit_lasso(
            Estimation_X,
            estimation_y,
            α,
            information_criteria,
            penalize_exogenous,
            components_indexes,
            penalty_factor;
            rm_average=true,
        )
    else
        penalty_factor = ones(size(Estimation_X, 2))
        if length(penalty_factor) != length(components_indexes["initial_states"])
            penalty_factor[components_indexes["initial_states"][1:end]] .= 0
        end
        penalty_factor[all_zero_idx] .= Inf
        coefs, _ = fit_lasso(
            Estimation_X,
            estimation_y,
            α,
            information_criteria,
            penalize_exogenous,
            components_indexes,
            penalty_factor;
            rm_average=false,
        )
    end

    #AdaLasso per component
    ts_penalty_factor =
        hasintercept ? zeros(size(Estimation_X, 2) - 1) : zeros(size(Estimation_X, 2))
    for key in keys(components_indexes)
        if key != "initial_states" && key != "μ1"
            component = components_indexes[key]
            if key in innovations_names
                κ = count(i -> i != 0, coefs[component]) < 1 ? 0 : std(coefs[component])
                if hasintercept
                    ts_penalty_factor[component .- 1] .= (1 / (κ + ϵ))
                else
                    ts_penalty_factor[component] .= (1 / (κ + ϵ))
                end
            else
                if hasintercept
                    ts_penalty_factor[component .- 1] = (1 ./ (abs.(coefs[component]) .+ ϵ))
                else
                    ts_penalty_factor[component] = (1 ./ (abs.(coefs[component]) .+ ϵ))
                end
            end
        end
    end

    if hasintercept
        if !penalize_initial_states
            ts_penalty_factor[components_indexes["initial_states"][2:end] .- 1] .= 0
        else
            nothing
        end
        ts_penalty_factor[all_zero_idx .- 1] .= Inf
    else
        if !penalize_initial_states
            ts_penalty_factor[components_indexes["initial_states"][1:end]] .= 0
        else
            nothing
        end
        ts_penalty_factor[all_zero_idx] .= Inf
    end

    return fit_lasso(
        Estimation_X,
        estimation_y,
        α,
        information_criteria,
        penalize_exogenous,
        components_indexes,
        ts_penalty_factor;
        rm_average=false,
    )
end

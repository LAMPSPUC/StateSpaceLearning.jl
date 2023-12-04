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
    fit_estimation_procedure(estimation_procedure::String, Estimation_X::Matrix{Tl}, estimation_y::Vector{Fl}, α::Float64,
                             hyperparameter_selection::String, components_indexes::Dict{String, Vector{Int64}},
                             adalasso_coef::Float64, select_exogenous::Bool)::Tuple{Vector{Float64}, Vector{Float64}} where {Tl, Fl}

    Fits the specified estimation procedure (currently Lasso or AdaLasso) to the provided data and returns coefficients and residuals.

    # Arguments
    - `estimation_procedure::String`: The chosen estimation procedure (either "Lasso" or "AdaLasso").
    - `Estimation_X::Matrix{Tl}`: Matrix of predictors for estimation.
    - `estimation_y::Vector{Fl}`: Vector of response values for estimation.
    - `α::Float64`: Elastic net control factor between ridge (α=0) and lasso (α=1) (default: 0.1).
    - `hyperparameter_selection::String`: Information Criteria method for hyperparameter selection (default: aic).
    - `components_indexes::Dict{String, Vector{Int64}}`: Dictionary containing indexes for different components.
    - `adalasso_coef::Float64`: AdaLasso adjustment coefficient (default: 0.1).
    - `select_exogenous::Bool`: Flag for selecting exogenous variables. When false the penalty factor for these variables will be set to 0.
    - `penalize_initial_states::Bool`: Flag to penalize initial states. When false the penalty factor for these variables will be set to 0.

    # Returns
    - `Tuple{Vector{Float64}, Vector{Float64}}`: Tuple containing coefficients and residuals of the fitted estimation procedure.

"""
function fit_estimation_procedure(estimation_procedure::String, Estimation_X::Matrix{Tl}, estimation_y::Vector{Fl}, α::Float64, 
                                    hyperparameter_selection::String, components_indexes::Dict{String, Vector{Int64}}, 
                                    adalasso_coef::Float64, select_exogenous::Bool, penalize_initial_states::Bool)::Tuple{Vector{Float64}, Vector{Float64}} where {Tl, Fl}

    estimation_arguments     = Dict("Lasso" => (Estimation_X, estimation_y, α, hyperparameter_selection, select_exogenous, components_indexes),
                                "AdaLasso" => (Estimation_X, estimation_y, α, hyperparameter_selection, 
                                               components_indexes, adalasso_coef, select_exogenous, penalize_initial_states))

    available_estimation = Dict("Lasso" => fit_lasso,
                                "AdaLasso" => fit_adalasso)

    return available_estimation[estimation_procedure](estimation_arguments[estimation_procedure]...)
end
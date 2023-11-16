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

function get_outlier_duplicate_columns(Estimation_X::Matrix{Tl}, components_indexes::Dict{String, Vector{Int64}}) where{Tl}
    o_indexes = components_indexes["o"]
    exogenous_indexes = components_indexes["Exogenous_X"]

    dummy_indexes = get_dummy_indexes(Estimation_X[:, exogenous_indexes])

    return o_indexes[dummy_indexes] .- 1
end

function fit_estimation_procedure(estimation_procedure::String, Estimation_X::Matrix{Tl}, estimation_y::Vector{Fl}, α::Float64, 
                                    hyperparameter_selection::String, components_indexes::Dict{String, Vector{Int64}}, 
                                    adalasso_coef::Float64, select_exogenous::Bool)::Tuple{Vector{Float64}, Vector{Float64}} where {Tl, Fl}

    estimation_arguments     = Dict("Lasso" => (Estimation_X, estimation_y, α, hyperparameter_selection, select_exogenous, components_indexes),
                                "AdaLasso" => (Estimation_X, estimation_y, α, hyperparameter_selection, 
                                               components_indexes, adalasso_coef, select_exogenous))

    available_estimation = Dict("Lasso" => fit_lasso,
                                "AdaLasso" => fit_adalasso)

    return available_estimation[estimation_procedure](estimation_arguments[estimation_procedure]...)
end
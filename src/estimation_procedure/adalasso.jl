function fit_adalasso(Estimation_X::Matrix{Tl}, estimation_y::Vector{Fl}, α::Float64, 
                        hyperparameter_selection::String, 
                        components_indexes::Dict{String, Vector{Int64}},
                        adalasso_coef::Float64, select_exogenous::Bool)::Tuple{Vector{Float64}, Vector{Float64}} where {Tl, Fl}

    penalty_factor = ones(size(Estimation_X, 2) - 1); penalty_factor[components_indexes["initial_states"][2:end] .- 1] .= 0
    coefs, _  = fit_lasso(Estimation_X, estimation_y, α, hyperparameter_selection, select_exogenous, components_indexes; penalty_factor = penalty_factor, intercept = false)

    #AdaLasso per component
    penalty_factor = zeros(size(Estimation_X, 2) - 1)
    for key in keys(components_indexes)
        if key != "initial_states" && key != "μ₁"
            component = components_indexes[key]
            if key != "Exogenous_X" && key != "o" && !(key in ["ν₁", "γ₁"])
                κ = count(i -> i == 0, coefs[component]) < 1 ? 0 : std(coefs[component])
                penalty_factor[component .- 1] .= (1 / (κ + adalasso_coef))
            else
                penalty_factor[component .- 1]  = (1 ./ (abs.(coefs[component]) .+ adalasso_coef))
            end
        end
    end 
    return fit_lasso(Estimation_X, estimation_y, α, hyperparameter_selection, select_exogenous, components_indexes; penalty_factor=penalty_factor)
end


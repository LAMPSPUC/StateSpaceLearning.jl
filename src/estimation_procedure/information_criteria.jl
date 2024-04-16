"""
    get_information(T::Int64, K::Int64, ϵ::Vector{Float64};
                    hyperparameter_selection::String = "bic", p::Int64 = 0)::Float64

    Calculates information criterion value based on the provided parameters and residuals.

    # Arguments
    - `T::Int64`: Number of observations.
    - `K::Int64`: Number of selected predictors.
    - `ϵ::Vector{Float64}`: Vector of residuals.
    - `hyperparameter_selection::String`: Method for hyperparameter selection (default: "bic").
    - `p::Int64`: Number of total predictors (default: 0).

    # Returns
    - `Float64`: Information criterion value.

"""
function get_information(T::Int64, K::Int64, ϵ::Vector{Float64}; hyperparameter_selection::String = "bic", p::Int64 = 0)::Float64
    if hyperparameter_selection == "bic"
        return T*log(var(ϵ)) + K*log(T)
    elseif hyperparameter_selection == "aic"
        return 2*K + T*log(var(ϵ))
    elseif hyperparameter_selection == "aicc"
        return 2*K + T*log(var(ϵ))  + ((2*K^2 +2*K)/(T - K - 1))
    elseif hyperparameter_selection == "EBIC"
        EBIC_comb_term = (K <= 1 || p == K) ? 0 : 2*(sum(log(j) for j in 1:p) - (sum(log(j) for j in 1:K) + sum(log(j) for j in 1:(p-K))))
        return T*log(var(ϵ)) + K*log(T) + EBIC_comb_term
    end
end
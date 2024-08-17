"""
    get_information(T::Int64, K::Int64, ε::Vector{Float64};
                    information_criteria::String = "bic", p::Int64 = 0)::Float64

    Calculates information criterion value based on the provided parameters and residuals.

    # Arguments
    - `T::Int64`: Number of observations.
    - `K::Int64`: Number of selected predictors.
    - `ε::Vector{Float64}`: Vector of residuals.
    - `information_criteria::String`: Method for hyperparameter selection (default: "aic").
    - `p::Int64`: Number of total predictors (default: 0).

    # Returns
    - `Float64`: Information criterion value.

"""
function get_information(T::Int64, K::Int64, ε::Vector{Float64}; information_criteria::String = "aic")::Float64
    if information_criteria == "bic"
        return T*log(var(ε)) + K*log(T)
    elseif information_criteria == "aic"
        return 2*K + T*log(var(ε))
    elseif information_criteria == "aicc"
        return 2*K + T*log(var(ε))  + ((2*K^2 +2*K)/(T - K - 1))
    end
end
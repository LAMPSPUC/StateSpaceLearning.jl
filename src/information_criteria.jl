"""
    get_information(T::Int, K::Int, ε::Vector{Float64};
                    information_criteria::String = "bic")::Float64

    Calculates information criterion value based on the provided parameters and residuals.

    # Arguments
    - `T::Int`: Number of observations.
    - `K::Int`: Number of selected predictors.
    - `ε::Vector{Float64}`: Vector of residuals.
    - `information_criteria::String`: Method for hyperparameter selection (default: "aic").

    # Returns
    - `Float64`: Information criterion value.

"""
function get_information(T::Int, K::Int, ε::Vector{Float64}; information_criteria::String = "aic")::Float64
    if information_criteria == "bic"
        return T*log(var(ε)) + K*log(T)
    elseif information_criteria == "aic"
        return 2*K + T*log(var(ε))
    elseif information_criteria == "aicc"
        return 2*K + T*log(var(ε))  + ((2*K^2 +2*K)/(T - K - 1))
    end
end
"""
    get_information(T::Int, K::Int, ε::Vector{Fl};
                    information_criteria::String = "aic",
                    p::Int = K, ebic_γ::AbstractFloat = 1.0)::AbstractFloat where Fl <: AbstractFloat

    Calculates information criterion value based on the provided parameters and residuals.

    # Arguments
    - `T::Int`: Number of observations.
    - `K::Int`: Number of selected predictors.
    - `ε::Vector{Fl}`: Vector of residuals.
    - `information_criteria::String`: Method for hyperparameter selection (default: "aic").
    - `p::Int`: Number of candidate predictors, only used by the "ebic" criterion (default: K, which makes it equivalent to "bic").
    - `ebic_γ::AbstractFloat`: Penalty parameter of the "ebic" criterion, in [0, 1] (default: 1.0).

    # Returns
    - `AbstractFloat`: Information criterion value.

"""
function get_information(
    T::Int,
    K::Int,
    ε::Vector{Fl};
    information_criteria::String="aic",
    p::Int=K,
    ebic_γ::AbstractFloat=1.0,
)::AbstractFloat where {Fl<:AbstractFloat}
    if information_criteria == "bic"
        return T * log(var(ε)) + K * log(T)
    elseif information_criteria == "ebic"
        return T * log(var(ε)) + K * log(T) + 2 * ebic_γ * log_binomial(p, K)
    elseif information_criteria == "aic"
        return 2 * K + T * log(var(ε))
    elseif information_criteria == "aicc"
        return 2 * K + T * log(var(ε)) + ((2 * K^2 + 2 * K) / (T - K - 1))
    else
        throw(
            ArgumentError(
                "information_criteria must be one of \"aic\", \"aicc\", \"bic\" or \"ebic\", got \"$(information_criteria)\"",
            ),
        )
    end
end

"""
    log_binomial(p::Int, K::Int)::AbstractFloat

    Calculates log(binomial(p, K)) in log space, avoiding the integer overflow of binomial(p, K) for large p.

    # Arguments
    - `p::Int`: Number of candidate predictors.
    - `K::Int`: Number of selected predictors.

    # Returns
    - `AbstractFloat`: Value of log(binomial(p, K)).

"""
function log_binomial(p::Int, K::Int)::AbstractFloat
    (K <= 0 || K >= p) && return 0.0
    return sum(log, (p - K + 1):p; init=0.0) - sum(log, 1:K; init=0.0)
end

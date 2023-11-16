mutable struct Output 
    model_type::String
    X::Matrix
    coefs::Vector
    ϵ::Vector
    fitted::Vector
    components::Dict
    residuals_variances::Dict
    s::Int64
    T::Int64
    outlier::Bool
    valid_indexes::Vector{Int64}
    stabilize_ζ::Int64
end
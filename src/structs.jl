"""
    mutable struct Output

    A mutable struct to store various components and results of a model estimation.

    # Fields
    - `model_type::String`: Type of the model.
    - `X::Matrix`: StateSpaceLearning Matrix data used in the model.
    - `coefs::Vector`: Coefficients obtained from the model.
    - `ϵ::Vector`: Residuals of the model.
    - `fitted::Vector`: Fitted values from the model.
    - `components::Dict`: Dictionary containing different components.
    - `residuals_variances::Dict`: Dictionary storing variances of residuals for different components.
    - `s::Int64`: Integer representing a parameter 's'.
    - `T::Int64`: Integer representing a parameter 'T'.
    - `outlier::Bool`: Boolean indicating the presence of outlier component.
    - `valid_indexes::Vector{Int64}`: Vector containing valid indexes of the time series.
    - `ζ_ω_threshold::Int64`: ζ_ω_threshold parameter.
    - `y::Vector{Fl}`: Vector of data.

"""
mutable struct Output 
    model_input::Dict
    X::Matrix
    coefs::Vector
    ϵ::Vector
    fitted::Vector
    components::Dict
    residuals_variances::Dict
    T::Int64
    outlier::Bool
    valid_indexes::Vector{Int64}
    ζ_ω_threshold::Int64
    y::Vector
end
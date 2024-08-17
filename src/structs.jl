"""
    mutable struct Output

    A mutable struct to store various components and results of a model estimation.

    # Fields
    - `model_input::Dict`: Dictionary containing the model input parameters.
    - `Create_X::Function`: Function used to create the StateSpaceLearning Matrix.
    - `X::Matrix`: StateSpaceLearning Matrix data used in the model.
    - `coefs::Vector`: Coefficients obtained from the model.
    - `ε::Vector`: Residuals of the model.
    - `fitted::Vector`: Fitted values from the model.
    - `components::Dict`: Dictionary containing different components.
    - `residuals_variances::Dict`: Dictionary storing variances of residuals for different components.
    - `T::Int64`: Integer representing a parameter 'T'.
    - `outlier::Bool`: Boolean indicating the presence of outlier component.
    - `valid_indexes::Vector{Int64}`: Vector containing valid indexes (non NaN) of the time series.
    - `ζ_ω_threshold::Int64`: ζ_ω_threshold parameter.
    - `y::Vector{Fl}`: Vector of data.

"""
mutable struct Output 
    model_input::Dict
    Create_X::Function
    X::Matrix
    coefs::Vector
    ε::Vector
    fitted::Vector
    components::Dict
    residuals_variances::Dict
    valid_indexes::Vector{Int64}
end
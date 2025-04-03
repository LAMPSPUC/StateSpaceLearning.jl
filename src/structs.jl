"""
    mutable struct Output

    A mutable struct to store various components and results of a model estimation.

    # Fields
    - `coefs::Vector`: Coefficients obtained from the model.
    - `ε::Vector`: Residuals of the model.
    - `fitted::Vector`: Fitted values from the model.
    - `residuals_variances::Dict`: Dictionary storing variances of residuals for different components.
    - `valid_indexes::Vector{Int}`: Vector containing valid indexes (non NaN) of the time series.
    - `components::Dict`: Dictionary containing components of the model.
"""
mutable struct Output
    coefs::Vector
    ε::Vector
    fitted::Vector
    residuals_variances::Dict
    valid_indexes::Vector{Int}
    components::Dict
    components_ts::Dict
end

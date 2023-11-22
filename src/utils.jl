"""
    build_components(X::Matrix{Tl}, coefs::Vector{Float64}, components_indexes::Dict{String, Vector{Int64}}) -> Dict where Tl
    
    Constructs components dict containing values, indexes and coefficients for each component.

    # Arguments
    - X::Matrix{Tl}: Input matrix.
    - coefs::Vector{Float64}: Coefficients.
    - components_indexes::Dict{String, Vector{Int64}}: Dictionary mapping component names to their indexes.

    # Returns
    - components::Dict: Dictionary containing components, each represented by a dictionary with keys:
        - "Coefs": Coefficients for the component.
        - "Indexes": Indexes associated with the component.
        - "Values": Values computed from `X` and component coefficients.

"""
function build_components(X::Matrix{Tl}, coefs::Vector{Float64}, components_indexes::Dict{String, Vector{Int64}})::Dict where Tl
    components = Dict()
    for key in keys(components_indexes)
        components[key] = Dict()
        components[key]["Coefs"]   = coefs[components_indexes[key]]
        components[key]["Indexes"] = components_indexes[key]
        components[key]["Values"]  = X[:, components_indexes[key]]*coefs[components_indexes[key]]
    end
    components["Exogenous_X"]["Selected"] = findall(i -> i != 0, components["Exogenous_X"]["Coefs"])
    return components
end

"""
    build_complete_variables(estimation_ϵ::Vector{Float64}, coefs::Vector{Float64}, X::Matrix{Tl}, valid_indexes::Vector{Int64}, T::Int64) -> Tuple{Vector{Float64}, Vector{Float64}} where Tl

    Builds complete residuals and fit in sample. Residuals will contain nan values for non valid indexes. Fit in Sample will be a vector of fitted values computed from input data and coefficients (non valid indexes will also be calculated via interpolation).

    # Arguments
    - `estimation_ϵ::Vector{Float64}`: Vector of estimation errors.
    - `coefs::Vector{Float64}`: Coefficients.
    - `X::Matrix{Tl}`: Input matrix.
    - `valid_indexes::Vector{Int64}`: Valid indexes.
    - `T::Int64`: Length of the original time series.

    # Returns
    - Tuple containing:
        - `ϵ::Vector{Float64}`: Vector containing NaN values filled with estimation errors at valid indexes.
        - `fitted::Vector{Float64}`: Vector of fitted values computed from input data and coefficients.

"""
function build_complete_variables(estimation_ϵ::Vector{Float64}, coefs::Vector{Float64}, X::Matrix{Tl}, valid_indexes::Vector{Int64}, T::Int64)::Tuple{Vector{Float64}, Vector{Float64}} where Tl
    ϵ      = fill(NaN, T); ϵ[valid_indexes] = estimation_ϵ
    fitted = X*coefs
    return ϵ, fitted
end
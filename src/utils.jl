"""
    build_components(X::Matrix{Tl}, coefs::Vector{Float64}, components_indexes::Dict{String, Vector{Int}}) -> Dict where Tl
    
    Constructs components dict containing values, indexes and coefficients for each component.

    # Arguments
    - X::Matrix{Tl}: Input matrix.
    - coefs::Vector{Float64}: Coefficients.
    - components_indexes::Dict{String, Vector{Int}}: Dictionary mapping component names to their indexes.

    # Returns
    - components::Dict: Dictionary containing components, each represented by a dictionary with keys:
        - "Coefs": Coefficients for the component.
        - "Indexes": Indexes associated with the component.
        - "Values": Values computed from `X` and component coefficients.

"""
function build_components(X::Matrix{Tl}, coefs::Vector{Float64}, components_indexes::Dict{String, Vector{Int}})::Dict where Tl
    components = Dict()
    for key in keys(components_indexes)
        components[key] = Dict()
        components[key]["Coefs"]   = coefs[components_indexes[key]]
        components[key]["Indexes"] = components_indexes[key]
        components[key]["Values"]  = X[:, components_indexes[key]]*coefs[components_indexes[key]]
    end
    if haskey(components, "Exogenous_X")
        components["Exogenous_X"]["Selected"] = findall(i -> i != 0, components["Exogenous_X"]["Coefs"])
    end
    return components
end

"""
get_fit_and_residuals(estimation_ε::Vector{Float64}, coefs::Vector{Float64}, X::Matrix{Tl}, valid_indexes::Vector{Int}, T::Int) -> Tuple{Vector{Float64}, Vector{Float64}} where Tl

    Builds complete residuals and fit in sample. Residuals will contain nan values for non valid indexes. Fit in Sample will be a vector of fitted values computed from input data and coefficients (non valid indexes will also be calculated via interpolation).

    # Arguments
    - `estimation_ε::Vector{Float64}`: Vector of estimation errors.
    - `coefs::Vector{Float64}`: Coefficients.
    - `X::Matrix{Tl}`: Input matrix.
    - `valid_indexes::Vector{Int}`: Valid indexes.
    - `T::Int`: Length of the original time series.

    # Returns
    - Tuple containing:
        - `ε::Vector{Float64}`: Vector containing NaN values filled with estimation errors at valid indexes.
        - `fitted::Vector{Float64}`: Vector of fitted values computed from input data and coefficients.

"""
function get_fit_and_residuals(estimation_ε::Vector{Float64}, coefs::Vector{Float64}, X::Matrix{Tl}, valid_indexes::Vector{Int}, T::Int)::Tuple{Vector{Float64}, Vector{Float64}} where Tl
    ε      = fill(NaN, T); ε[valid_indexes] = estimation_ε
    fitted = X*coefs
    return ε, fitted
end

"""
o_size(T::Int)::Int

    Calculates the size of outlier matrix based on the input T.

    # Arguments
    - `T::Int`: Length of the original time series.

    # Returns
    - `Int`: Size of o calculated from T.

"""
o_size(T::Int)::Int = T

"""
create_o_matrix(T::Int, steps_ahead::Int)::Matrix

    Creates a matrix of outliers based on the input sizes, and the desired steps ahead (this is necessary for the forecast function).

    # Arguments
    - `T::Int`: Length of the original time series.
    - `steps_ahead::Int`: Number of steps ahead (for estimation purposes this should be set at 0).
    
    # Returns
    - `Matrix`: Matrix of outliers constructed based on the input sizes.

"""
function create_o_matrix(T::Int, steps_ahead::Int)::Matrix
    return vcat(Matrix(1.0 * I, T, T), zeros(steps_ahead, T))
end

"""
handle_missing_values(X::Matrix{Tl}, y::Vector{Fl}) -> Tuple{Vector{Fl}, Matrix{Tl}} where {Tl, Fl}

    Removes missing values from input data and returns the time series and matrix without missing values.

    # Arguments
    - `X::Matrix{Tl}`: Input matrix.
    - `y::Vector{Fl}`: Time series.

    # Returns
    - Tuple containing:
        - `y::Vector{Fl}`: Time series without missing values.
        - `X::Matrix{Tl}`: Input matrix without missing values.
        - `valid_indexes::Vector{Int}`: Vector containing valid indexes of the time series.
"""
function handle_missing_values(X::Matrix{Tl}, y::Vector{Fl})::Tuple{Vector{Fl}, Matrix{Tl}, Vector{Int}} where {Tl, Fl}

    invalid_indexes = unique(vcat([i[1] for i in findall(i -> any(isnan, i), X)], findall(i -> isnan(i), y)))
    valid_indexes   = setdiff(1:length(y), invalid_indexes)

    return y[valid_indexes], X[valid_indexes, :], valid_indexes
end

"""
has_intercept(X::Matrix{Tl})::Bool where Tl

    Checks if the input matrix has a constant column (intercept).

    # Arguments
    - `X::Matrix{Tl}`: Input matrix.

    # Returns
    - `Bool`: True if the input matrix has a constant column, false otherwise.
"""
function has_intercept(X::Matrix{Tl})::Bool where Tl
    return any([all(X[:, i] .== 1) for i in 1:size(X, 2)])
end

"""
fill_innovation_coefs(component::String, output::Output)::Vector{Float64}

    Build the innovation coefficients for a given component with same length as the original time series and coefficients attributed to the first observation they are associated with.

    # Arguments
    - `component::String`: Component name.
    - `output::Output`: Output object obtained from model fitting.

    # Returns
    - `Vector{Float64}`: Vector containing innovation coefficients for the given component.
"""
function fill_innovation_coefs(T::Int, component::String, output::Output)::Vector{Float64}
    inov_comp = zeros(T)
    for (i, idx) in enumerate(output.components[component]["Indexes"])
        inov_comp[findfirst(i -> i != 0, output.X[:, idx])] = output.components[component]["Coefs"][i]
    end
    return inov_comp
end
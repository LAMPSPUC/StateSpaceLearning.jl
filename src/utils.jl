"""
    build_components(
    X::Matrix{Tl}, coefs::Vector{Fl}, components_indexes::Dict{String,Vector{Int}}
)::Dict where {Fl <: AbstractFloat, Tl <: AbstractFloat}

    Constructs components dict containing values, indexes and coefficients for each component.

    # Arguments
    - X::Matrix{Tl}: Input matrix.
    - coefs::Vector{Fl}: Coefficients.
    - components_indexes::Dict{String, Vector{Int}}: Dictionary mapping component names to their indexes.

    # Returns
    - components::Dict: Dictionary containing components, each represented by a dictionary with keys:
        - "Coefs": Coefficients for the component.
        - "Indexes": Indexes associated with the component.
        - "Values": Values computed from `X` and component coefficients.

"""
function build_components(
    X::Matrix{Tl}, coefs::Vector{Fl}, components_indexes::Dict{String,Vector{Int}}
)::Dict where {Fl<:AbstractFloat,Tl<:AbstractFloat}
    components = Dict()
    for key in keys(components_indexes)
        components[key] = Dict()
        components[key]["Coefs"] = coefs[components_indexes[key]]
        components[key]["Indexes"] = components_indexes[key]
        components[key]["Values"] =
            X[:, components_indexes[key]] * coefs[components_indexes[key]]
    end
    if haskey(components, "exog")
        components["exog"]["Selected"] = findall(i -> i != 0, components["exog"]["Coefs"])
    end
    return components
end

"""
get_fit_and_residuals(
    estimation_ε::Vector{Fl},
    coefs::Vector{Pl},
    X::Matrix{Tl},
    valid_indexes::Vector{Int},
    T::Int,
)::Tuple{Vector{AbstractFloat},Vector{AbstractFloat}} where {Fl <: AbstractFloat, Pl <: AbstractFloat, Tl <: AbstractFloat}

    Builds complete residuals and fit in sample. Residuals will contain nan values for non valid indexes. Fit in Sample will be a vector of fitted values computed from input data and coefficients (non valid indexes will also be calculated via interpolation).

    # Arguments
    - `estimation_ε::Vector{Fl}`: Vector of estimation errors.
    - `coefs::Vector{Pl}`: Coefficients.
    - `X::Matrix{Tl}`: Input matrix.
    - `valid_indexes::Vector{Int}`: Valid indexes.
    - `T::Int`: Length of the original time series.

    # Returns
    - Tuple containing:
        - `ε::Vector{AbstractFloat}`: Vector containing NaN values filled with estimation errors at valid indexes.
        - `fitted::Vector{AbstractFloat}`: Vector of fitted values computed from input data and coefficients.

"""
function get_fit_and_residuals(
    estimation_ε::Vector{Fl},
    coefs::Vector{Pl},
    X::Matrix{Tl},
    valid_indexes::Vector{Int},
    T::Int,
)::Tuple{
    Vector{AbstractFloat},Vector{AbstractFloat}
} where {Fl<:AbstractFloat,Pl<:AbstractFloat,Tl<:AbstractFloat}
    ε = fill(NaN, T)
    ε[valid_indexes] = estimation_ε
    fitted = X * coefs
    return ε, fitted
end

"""
    handle_missing_values(
    X::Matrix{Tl}, y::Vector{Fl}
)::Tuple{Vector{Fl},Matrix{Fl},Vector{Int}} where {Fl <: AbstractFloat, Tl <: AbstractFloat}

    Removes missing values from input data and returns the time series and matrix without missing values.

    # Arguments
    - `X::Matrix{Fl}`: Input matrix.
    - `y::Vector{Fl}`: Time series.

    # Returns
    - Tuple containing:
        - `y::Vector{Fl}`: Time series without missing values.
        - `X::Matrix{Fl}`: Input matrix without missing values.
        - `valid_indexes::Vector{Int}`: Vector containing valid indexes of the time series.
"""
function handle_missing_values(
    X::Matrix{Tl}, y::Vector{Fl}
)::Tuple{Vector{Fl},Matrix{Fl},Vector{Int}} where {Fl<:AbstractFloat,Tl<:AbstractFloat}
    invalid_indexes = unique(
        vcat([i[1] for i in findall(i -> any(isnan, i), X)], findall(i -> isnan(i), y))
    )
    valid_indexes = setdiff(1:length(y), invalid_indexes)

    return y[valid_indexes], X[valid_indexes, :], valid_indexes
end

"""
has_intercept(X::Matrix{Fl})::Bool where Fl <: AbstractFloat

    Checks if the input matrix has a constant column (intercept).

    # Arguments
    - `X::Matrix{Fl}`: Input matrix.

    # Returns
    - `Bool`: True if the input matrix has a constant column, false otherwise.
"""
function has_intercept(X::Matrix{Fl})::Bool where {Fl<:AbstractFloat}
    return any([all(X[:, i] .== 1) for i in 1:size(X, 2)])
end

"""
get_stochastic_values(estimated_stochastic::Vector{Fl}, steps_ahead::Int, T::Int, start_idx::Int, final_idx::Int, seasonal_innovation_simulation::Int)::Vector{AbstractFloat} where {Fl<:AbstractFloat}

    Generates stochastic seasonal values for a given time series.

    # Arguments
    - `estimated_stochastic::Vector{Fl}`: Vector of estimated stochastic terms.
    - `steps_ahead::Int`: Number of steps ahead to generate.
    - `T::Int`: Length of the time series.
    - `start_idx::Int`: Starting index of the time series.
    - `final_idx::Int`: Final index of the time series.
    - `seasonal_innovation_simulation::Int`: Seasonal innovation simulation.

    # Returns
    - `Vector{AbstractFloat}`: Vector of stochastic seasonal values.
"""
function get_stochastic_values(
    estimated_stochastic::Vector{Fl},
    steps_ahead::Int,
    T::Int,
    start_idx::Int,
    final_idx::Int,
    seasonal_innovation_simulation::Int,
    N_scenarios::Int,
)::Matrix{AbstractFloat} where {Fl<:AbstractFloat}
    if seasonal_innovation_simulation != 0
        stochastic_term = Matrix{AbstractFloat}(undef, steps_ahead, N_scenarios)
        for t in 1:steps_ahead

            # Generate potential seasonal indices
            seasonal_indices =
                ((T + t) % seasonal_innovation_simulation):seasonal_innovation_simulation:T

            # Filter indices to be within the valid range
            valid_indices = filter(idx -> start_idx <= idx <= final_idx, seasonal_indices)

            # Sample with randomness and sign flip
            if !isempty(estimated_stochastic[valid_indices])
                stochastic_term[t, :] =
                    rand(estimated_stochastic[valid_indices], N_scenarios) .*
                    rand([1, -1], N_scenarios)
            else
                stochastic_term[t, :] .= 0.0
            end
        end
    else
        stochastic_term =
            rand(estimated_stochastic, steps_ahead, N_scenarios) .*
            rand([1, -1], steps_ahead, N_scenarios)
    end

    return stochastic_term
end

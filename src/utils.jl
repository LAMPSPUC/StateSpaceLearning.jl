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
    if haskey(components, "Exogenous_X")
        components["Exogenous_X"]["Selected"] = findall(
            i -> i != 0, components["Exogenous_X"]["Coefs"]
        )
    end
    return components
end

"""
    build_components(
    X::Matrix{Tl}, coefs::Vector{Vector{Fl}}, components_indexes::Dict{String,Vector{Int}}
)::Vector{Dict} where {Fl <: AbstractFloat, Tl <: AbstractFloat}

    Constructs components dict containing values, indexes and coefficients for each component.

    # Arguments
    - X::Matrix{Fl}: Input matrix.
    - coefs::Vector{Vector{Fl}}: Coefficients for each time series.
    - components_indexes::Dict{String, Vector{Int}}: Dictionary mapping component names to their indexes.

    # Returns
    - components::Vector{Dict}: Dictionary containing components, each represented by a dictionary with keys:
        - "Coefs": Coefficients for the component.
        - "Indexes": Indexes associated with the component.
        - "Values": Values computed from `X` and component coefficients.

"""
function build_components(
    X::Matrix{Tl}, coefs::Vector{Vector{Fl}}, components_indexes::Dict{String,Vector{Int}}
)::Vector{Dict} where {Fl<:AbstractFloat,Tl<:AbstractFloat}
    components_vec = Dict[]
    for coef_el in coefs
        components = Dict()
        for key in keys(components_indexes)
            components[key] = Dict()
            components[key]["Coefs"] = coef_el[components_indexes[key]]
            components[key]["Indexes"] = components_indexes[key]
            components[key]["Values"] =
                X[:, components_indexes[key]] * coef_el[components_indexes[key]]
        end
        if haskey(components, "Exogenous_X")
            components["Exogenous_X"]["Selected"] = findall(
                i -> i != 0, components["Exogenous_X"]["Coefs"]
            )
        end
        push!(components_vec, components)
    end
    return components_vec
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
get_fit_and_residuals(
    estimation_ε::Vector{Vector{Fl}},
    coefs::Vector{Vector{Pl}},
    X::Matrix{Tl},
    valid_indexes::Vector{Int},
    T::Int,
)::Tuple{Vector{Vector{AbstractFloat}},Vector{Vector{AbstractFloat}}} where {Fl <: AbstractFloat, Pl <: AbstractFloat, Tl <: AbstractFloat}

    Builds complete residuals and fit in sample. Residuals will contain nan values for non valid indexes. Fit in Sample will be a vector of fitted values computed from input data and coefficients (non valid indexes will also be calculated via interpolation).

    # Arguments
    - `estimation_ε::Vector{Vector{Fl}}`: Vector of estimation errors.
    - `coefs::Vector{Vector{Pl}}`: Coefficients.
    - `X::Matrix{Tl}`: Input matrix.
    - `valid_indexes::Vector{Int}`: Valid indexes.
    - `T::Int`: Length of the original time series.

    # Returns
    - Tuple containing:
        - `ε::Vector{AbstractFloat}`: Vector containing NaN values filled with estimation errors at valid indexes.
        - `fitted::Vector{AbstractFloat}`: Vector of fitted values computed from input data and coefficients.

"""
function get_fit_and_residuals(
    estimation_ε::Vector{Vector{Fl}},
    coefs::Vector{Vector{Pl}},
    X::Matrix{Tl},
    valid_indexes::Vector{Int},
    T::Int,
)::Tuple{
    Vector{Vector{AbstractFloat}},Vector{Vector{AbstractFloat}}
} where {Fl<:AbstractFloat,Pl<:AbstractFloat,Tl<:AbstractFloat}
    ε_vec = Vector{AbstractFloat}[]
    fitted_vec = Vector{AbstractFloat}[]

    for i in eachindex(coefs)
        ε = fill(NaN, T)
        ε[valid_indexes] = estimation_ε[i]
        fitted = X * coefs[i]
        push!(ε_vec, ε)
        push!(fitted_vec, fitted)
    end
    return ε_vec, fitted_vec
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
handle_missing_values(
    X::Matrix{Tl}, y::Matrix{Fl}
)::Tuple{Matrix{Fl},Matrix{Fl},Vector{Int}} where {Fl <: AbstractFloat, Tl <: AbstractFloat}

    Removes missing values from input data and returns the time series and matrix without missing values.

    # Arguments
    - `X::Matrix{Tl}`: Input matrix.
    - `y::Matrix{Fl}`: Time series.

    # Returns
    - Tuple containing:
        - `y::Vector{Fl}`: Time series without missing values.
        - `X::Matrix{Fl}`: Input matrix without missing values.
        - `valid_indexes::Vector{Int}`: Vector containing valid indexes of the time series.
"""
function handle_missing_values(
    X::Matrix{Tl}, y::Matrix{Fl}
)::Tuple{Matrix{Fl},Matrix{Fl},Vector{Int}} where {Fl<:AbstractFloat,Tl<:AbstractFloat}
    invalid_cartesian_indexes = unique(
        vcat([i[1] for i in findall(i -> any(isnan, i), X)], findall(i -> isnan(i), y))
    )

    invalid_indexes = Int[]
    for i in invalid_cartesian_indexes
        if !(i[1] in invalid_indexes)
            push!(invalid_indexes, i[1])
        end
    end

    valid_indexes = setdiff(1:size(y, 1), invalid_indexes)

    return y[valid_indexes, :], X[valid_indexes, :], valid_indexes
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
fill_innovation_coefs(model::StructuralModel, T::Int, component::String, valid_indexes::Vector{Int})::Vector{AbstractFloat}

    Build the innovation coefficients for a given component with same length as the original time series and coefficients attributed to the first observation they are associated with.

    # Arguments
    - `model::StructuralModel`: Structural model.
    - `T::Int`: Length of the original time series.
    - `component::String`: Component name.
    - `valid_indexes::Vector{Int}`: Valid Indexes in the time series

    # Returns
    - `Union{Vector{AbstractFloat}, Matrix{AbstractFloat}}`: Vector or matrix containing innovation coefficients for the given component.
"""
function fill_innovation_coefs(
    model::StructuralModel, component::String, valid_indexes::Vector{Int}
)::Union{Vector,Matrix}
    T = length(model.y)
    if typeof(model.output) == Output
        inov_comp = zeros(T)
        for (i, idx) in enumerate(model.output.components[component]["Indexes"])
            inov_comp[findfirst(i -> i != 0, model.X[:, idx])] = model.output.components[component]["Coefs"][i]
        end
        inov_comp = inov_comp[valid_indexes]
    else
        inov_comp = zeros(T, length(model.output))
        for j in eachindex(model.output)
            for (i, idx) in enumerate(model.output[j].components[component]["Indexes"])
                inov_comp[findfirst(i -> i != 0, model.X[:, idx]), j] = model.output[j].components[component]["Coefs"][i]
            end
        end
        inov_comp = inov_comp[valid_indexes, :]
    end
    return inov_comp
end

"""
fill_simulation!(simulation::Matrix{Tl}, MV_dist_vec::Vector{MvNormal}, o_noises::Matrix{Fl}, simulation_X::Matrix{Pl}) where {Fl <: AbstractFloat, Pl <: AbstractFloat, Tl <: AbstractFloat}

    Fill the simulation matrix with the generated values

    # Arguments
    - `simulation::Matrix{Tl}`: Matrix to be filled with simulated values.
    - `MV_dist_vec::Vector{MvNormal}`: Vector of MvNormal distributions.
    - `o_noises::Matrix{Fl}`: Matrix of outliers.
    - `simulation_X::Matrix{Pl}`: Matrix of simulation coefficients.
"""
function fill_simulation!(
    simulation::Matrix{Tl},
    MV_dist_vec::Vector{MvNormal},
    o_noises::Matrix{Fl},
    simulation_X::Matrix{Pl},
) where {Fl<:AbstractFloat,Pl<:AbstractFloat,Tl<:AbstractFloat}
    steps_ahead, N_scenarios = size(simulation)
    for s in 1:N_scenarios
        sim_coefs = ones(size(simulation_X, 2)) .* NaN

        for i in 1:steps_ahead
            rand_inovs = rand(MV_dist_vec[i])

            for comp in eachindex(rand_inovs)
                sim_coefs[i + (comp - 1) * steps_ahead] = rand_inovs[comp]
            end
        end

        simulation[:, s] += (simulation_X * sim_coefs + o_noises[:, s])
    end
end

"""
fill_simulation!(simulation::Vector{Matrix{Tl}}, MV_dist_vec::Vector{MvNormal}, o_noises::Vector{Matrix{Fl}}, simulation_X::Matrix{Pl}, N_innovations::Int) where {Fl <: AbstractFloat, Pl <: AbstractFloat, Tl <: AbstractFloat}

    Fill the simulation matrix with the generated values

    # Arguments
    - `simulation::Vector{Matrix{Tl}}`: Vector of matrices to be filled with simulated values.
    - `MV_dist_vec::Vector{MvNormal}`: Vector of MvNormal distributions.
    - `o_noises::Vector{Matrix{Fl}}`: Vector of matrices of outliers.
    - `simulation_X::Matrix{Pl}`: Matrix of simulation coefficients.
    - `N_innovations::Int`: Number of innovations.
"""
function fill_simulation!(
    simulation::Vector{Matrix{Tl}},
    MV_dist_vec::Vector{MvNormal},
    o_noises::Vector{Matrix{Fl}},
    simulation_X::Matrix{Pl},
    N_innovations::Int,
) where {Fl<:AbstractFloat,Pl<:AbstractFloat,Tl<:AbstractFloat}
    steps_ahead, N_scenarios = size(simulation[1])
    for j in eachindex(simulation)
        for s in 1:N_scenarios
            sim_coefs = ones(size(simulation_X, 2)) .* NaN

            for i in 1:steps_ahead
                rand_inovs = rand(MV_dist_vec[i])[j:N_innovations:end]

                for comp in eachindex(rand_inovs)
                    sim_coefs[i + (comp - 1) * steps_ahead] = rand_inovs[comp]
                end
            end

            simulation[j][:, s] += (simulation_X * sim_coefs + o_noises[j][:, s])
        end
    end
end

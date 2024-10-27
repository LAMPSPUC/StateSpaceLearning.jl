mutable struct StructuralModel <: StateSpaceLearningModel
    y::Vector
    X::Matrix
    level::Bool
    stochastic_level::Bool
    trend::Bool
    stochastic_trend::Bool
    seasonal::Bool
    stochastic_seasonal::Bool
    freq_seasonal::Int
    outlier::Bool
    ζ_ω_threshold::Int
    n_exogenous::Int
    output::Union{Output,Nothing}

    function StructuralModel(
        y::Vector{Fl};
        level::Bool=true,
        stochastic_level::Bool=true,
        trend::Bool=true,
        stochastic_trend::Bool=true,
        seasonal::Bool=true,
        stochastic_seasonal::Bool=true,
        freq_seasonal::Int=12,
        outlier::Bool=true,
        ζ_ω_threshold::Int=12,
        Exogenous_X::Matrix{Fl}=zeros(length(y), 0),
    ) where {Fl}
        n_exogenous = size(Exogenous_X, 2)
        @assert !has_intercept(Exogenous_X) "Exogenous matrix must not have an intercept column"
        @assert seasonal ? length(y) > freq_seasonal : true "Time series must be longer than the seasonal period"

        X = create_X(
            level,
            stochastic_level,
            trend,
            stochastic_trend,
            seasonal,
            stochastic_seasonal,
            freq_seasonal,
            outlier,
            ζ_ω_threshold,
            Exogenous_X,
        )

        return new(
            y,
            X,
            level,
            stochastic_level,
            trend,
            stochastic_trend,
            seasonal,
            stochastic_seasonal,
            freq_seasonal,
            outlier,
            ζ_ω_threshold,
            n_exogenous,
            nothing,
        )
    end
end

"""
    ξ_size(T::Int)::Int

    Calculates the size of ξ innovation matrix based on the input T.

    # Arguments
    - `T::Int`: Length of the original time series.

    # Returns
    - `Int`: Size of ξ calculated from T.

"""
ξ_size(T::Int)::Int = T - 2

"""
ζ_size(T::Int, ζ_ω_threshold::Int)::Int

    Calculates the size of ζ innovation matrix based on the input T.

    # Arguments
    - `T::Int`: Length of the original time series.
    - `ζ_ω_threshold::Int`: Stabilize parameter ζ.

    # Returns
    - `Int`: Size of ζ calculated from T.

"""
ζ_size(T::Int, ζ_ω_threshold::Int)::Int = T - ζ_ω_threshold - 2

"""
ω_size(T::Int, s::Int)::Int

    Calculates the size of ω innovation matrix based on the input T.

    # Arguments
    - `T::Int`: Length of the original time series.
    - `s::Int`: Seasonal period.

    # Returns
    - `Int`: Size of ω calculated from T.

"""
ω_size(T::Int, s::Int, ζ_ω_threshold::Int)::Int = T - ζ_ω_threshold - s + 1

"""
    create_ξ(T::Int, steps_ahead::Int)::Matrix

    Creates a matrix of innovations ξ based on the input sizes, and the desired steps ahead (this is necessary for the forecast function)

    # Arguments
    - `T::Int`: Length of the original time series.
    - `steps_ahead::Int`: Number of steps ahead (for estimation purposes this should be set at 0).

    # Returns
    - `Matrix`: Matrix of innovations ξ constructed based on the input sizes.

"""
function create_ξ(T::Int, steps_ahead::Int)::Matrix
    ξ_matrix = Matrix{Float64}(undef, T + steps_ahead, T - 1)
    for t in 1:(T + steps_ahead)
        ξ_matrix[t, :] = t < T ? vcat(ones(t - 1), zeros(T - t)) : ones(T - 1)
    end

    return ξ_matrix[:, 1:(end - 1)]
end

"""
create_ζ(T::Int, steps_ahead::Int, ζ_ω_threshold::Int)::Matrix

    Creates a matrix of innovations ζ based on the input sizes, and the desired steps ahead (this is necessary for the forecast function).

    # Arguments
    - `T::Int`: Length of the original time series.
    - `steps_ahead::Int`: Number of steps ahead (for estimation purposes this should be set at 0).
    - `ζ_ω_threshold::Int`: Stabilize parameter ζ.

    # Returns
    - `Matrix`: Matrix of innovations ζ constructed based on the input sizes.

"""
function create_ζ(T::Int, steps_ahead::Int, ζ_ω_threshold::Int)::Matrix
    ζ_matrix = Matrix{Float64}(undef, T + steps_ahead, T - 2)
    for t in 1:(T + steps_ahead)
        ζ_matrix[t, :] = if t < T
            vcat(collect((t - 2):-1:1), zeros(T - 2 - length(collect((t - 2):-1:1))))
        else
            collect((t - 2):-1:(t - T + 1))
        end
    end
    return ζ_matrix[:, 1:(end - ζ_ω_threshold)]
end

"""
create_ω(T::Int, s::Int, steps_ahead::Int)::Matrix

    Creates a matrix of innovations ω based on the input sizes, and the desired steps ahead (this is necessary for the forecast function).

    # Arguments
    - `T::Int`: Length of the original time series.
    - `freq_seasonal::Int`: Seasonal period.
    - `steps_ahead::Int`: Number of steps ahead (for estimation purposes this should be set at 0).
    
    # Returns
    - `Matrix`: Matrix of innovations ω constructed based on the input sizes.

"""
function create_ω(T::Int, freq_seasonal::Int, steps_ahead::Int, ζ_ω_threshold::Int)::Matrix
    ω_matrix_size = ω_size(T, freq_seasonal, ζ_ω_threshold) + ζ_ω_threshold
    ω_matrix = zeros(T + steps_ahead, ω_matrix_size)
    for t in (freq_seasonal + 1):(T + steps_ahead)
        ωₜ_coefs = zeros(ω_matrix_size)
        Mₜ = Int(floor((t - 1) / freq_seasonal))
        lag₁ = [t - j * freq_seasonal for j in 0:(Mₜ - 1)]
        lag₂ = [t - j * freq_seasonal - 1 for j in 0:(Mₜ - 1)]
        ωₜ_coefs[lag₁[lag₁ .<= ω_matrix_size + (freq_seasonal - 1)] .- (freq_seasonal - 1)] .=
            1
        ωₜ_coefs[lag₂[0 .< lag₂ .<= ω_matrix_size + (freq_seasonal - 1)] .- (freq_seasonal - 1)] .=
            -1
        ω_matrix[t, :] = ωₜ_coefs
    end
    return ω_matrix[:, 1:(end - ζ_ω_threshold)]
end

"""
    create_initial_states_Matrix(T::Int, s::Int, steps_ahead::Int, level::Bool, trend::Bool, seasonal::Bool)::Matrix

    Creates an initial states matrix based on the input parameters.

    # Arguments
    - `T::Int`: Length of the original time series.
    - `freq_seasonal::Int`: Seasonal period.
    - `steps_ahead::Int`: Number of steps ahead.
    - `level::Bool`: Flag for considering level component.
    - `trend::Bool`: Flag for considering trend component.
    - `seasonal::Bool`: Flag for considering seasonal component.

    # Returns
    - `Matrix`: Initial states matrix constructed based on the input parameters.

"""
function create_initial_states_Matrix(
    T::Int, freq_seasonal::Int, steps_ahead::Int, level::Bool, trend::Bool, seasonal::Bool
)::Matrix
    initial_states_matrix = zeros(T + steps_ahead, 0)
    if level
        initial_states_matrix = hcat(initial_states_matrix, ones(T + steps_ahead, 1))
    else
        nothing
    end
    if trend
        initial_states_matrix = hcat(
            initial_states_matrix, vcat([0], collect(1:(T + steps_ahead - 1)))
        )
    else
        nothing
    end

    if seasonal
        γ1_matrix = zeros(T + steps_ahead, freq_seasonal)
        for t in 1:(T + steps_ahead)
            γ1_matrix[t, t % freq_seasonal == 0 ? freq_seasonal : t % freq_seasonal] = 1.0
        end
        return hcat(initial_states_matrix, γ1_matrix)
    end

    return initial_states_matrix
end

"""
create_X(level::Bool, stochastic_level::Bool, trend::Bool, stochastic_trend::Bool,
                  seasonal::Bool, stochastic_seasonal::Bool, freq_seasonal::Int, outlier::Bool, ζ_ω_threshold::Int, Exogenous_X::Matrix{Fl},
                  steps_ahead::Int=0, Exogenous_Forecast::Matrix{Fl}=zeros(steps_ahead, size(Exogenous_X, 2))) where Fl

    Creates the StateSpaceLearning matrix X based on the model type and input parameters.

    # Arguments
    - `level::Bool`: Flag for considering level component.
    - `stochastic_level::Bool`: Flag for considering stochastic level component.
    - `trend::Bool`: Flag for considering trend component.
    - `stochastic_trend::Bool`: Flag for considering stochastic trend component.
    - `seasonal::Bool`: Flag for considering seasonal component.
    - `stochastic_seasonal::Bool`: Flag for considering stochastic seasonal component.
    - `freq_seasonal::Int`: Seasonal period.
    - `outlier::Bool`: Flag for considering outlier component.
    - `ζ_ω_threshold::Int`: Stabilize parameter ζ.
    - `Exogenous_X::Matrix{Fl}`: Exogenous variables matrix.
    - `steps_ahead::Int`: Number of steps ahead (default: 0).
    - `Exogenous_Forecast::Matrix{Fl}`: Exogenous variables forecast matrix (default: zeros).

    # Returns
    - `Matrix`: StateSpaceLearning matrix X constructed based on the input parameters.
"""
function create_X(
    level::Bool,
    stochastic_level::Bool,
    trend::Bool,
    stochastic_trend::Bool,
    seasonal::Bool,
    stochastic_seasonal::Bool,
    freq_seasonal::Int,
    outlier::Bool,
    ζ_ω_threshold::Int,
    Exogenous_X::Matrix{Fl},
    steps_ahead::Int=0,
    Exogenous_Forecast::Matrix{Fl}=zeros(steps_ahead, size(Exogenous_X, 2)),
) where {Fl}
    T = size(Exogenous_X, 1)

    ξ_matrix = stochastic_level ? create_ξ(T, steps_ahead) : zeros(T + steps_ahead, 0)
    ζ_matrix = if stochastic_trend
        create_ζ(T, steps_ahead, ζ_ω_threshold)
    else
        zeros(T + steps_ahead, 0)
    end
    ω_matrix = if stochastic_seasonal
        create_ω(T, freq_seasonal, steps_ahead, ζ_ω_threshold)
    else
        zeros(T + steps_ahead, 0)
    end
    o_matrix = outlier ? create_o_matrix(T, steps_ahead) : zeros(T + steps_ahead, 0)

    initial_states_matrix = create_initial_states_Matrix(
        T, freq_seasonal, steps_ahead, level, trend, seasonal
    )
    return hcat(
        initial_states_matrix,
        ξ_matrix,
        ζ_matrix,
        ω_matrix,
        o_matrix,
        vcat(Exogenous_X, Exogenous_Forecast),
    )
end

"""
function get_components_indexes(model::StructuralModel)::Dict where Fl

    Generates indexes dict for different components based on the model type and input parameters.

    # Arguments
    - `model::StructuralModel`: StructuralModel object.

    # Returns
    - `Dict`: Dictionary containing the corresponding indexes for each component of the model.

"""
function get_components_indexes(model::StructuralModel)::Dict
    T = length(model.y)

    FINAL_INDEX = 0

    if model.level
        μ1_indexes = [1]
        initial_states_indexes = [1]
        FINAL_INDEX += length(μ1_indexes)
    else
        μ1_indexes = Int[]
        initial_states_indexes = Int[]
    end

    if model.trend
        ν1_indexes = [2]
        initial_states_indexes = vcat(initial_states_indexes, ν1_indexes)
        FINAL_INDEX += length(ν1_indexes)
    else
        ν1_indexes = Int[]
    end

    if model.seasonal
        γ1_indexes = collect((FINAL_INDEX + 1):(FINAL_INDEX + model.freq_seasonal))
        initial_states_indexes = vcat(initial_states_indexes, γ1_indexes)
        FINAL_INDEX += length(γ1_indexes)
    else
        γ1_indexes = Int[]
    end

    if model.stochastic_level
        ξ_indexes = collect((FINAL_INDEX + 1):(FINAL_INDEX + ξ_size(T)))
        FINAL_INDEX += length(ξ_indexes)
    else
        ξ_indexes = Int[]
    end

    if model.stochastic_trend
        ζ_indexes = collect(
            (FINAL_INDEX + 1):(FINAL_INDEX + ζ_size(T, model.ζ_ω_threshold))
        )
        FINAL_INDEX += length(ζ_indexes)
    else
        ζ_indexes = Int[]
    end

    if model.stochastic_seasonal
        ω_indexes = collect(
            (FINAL_INDEX + 1):(FINAL_INDEX + ω_size(
                T, model.freq_seasonal, model.ζ_ω_threshold
            )),
        )
        FINAL_INDEX += length(ω_indexes)
    else
        ω_indexes = Int[]
    end

    if model.outlier
        o_indexes = collect((FINAL_INDEX + 1):(FINAL_INDEX + o_size(T)))
        FINAL_INDEX += length(o_indexes)
    else
        o_indexes = Int[]
    end

    exogenous_indexes = collect((FINAL_INDEX + 1):(FINAL_INDEX + model.n_exogenous))

    return Dict(
        "μ1" => μ1_indexes,
        "ν1" => ν1_indexes,
        "γ1" => γ1_indexes,
        "ξ" => ξ_indexes,
        "ζ" => ζ_indexes,
        "ω" => ω_indexes,
        "o" => o_indexes,
        "Exogenous_X" => exogenous_indexes,
        "initial_states" => initial_states_indexes,
    )
end

"""
    function get_variances(model::StructuralModel, ε::Vector{Fl}, coefs::Vector{Fl}, components_indexes::Dict{String, Vector{Int}})::Dict where Fl

    Calculates variances for each innovation component and for the residuals.

    # Arguments
    - `model::StructuralModel`: StructuralModel object.
    - `ε::Vector{Fl}`: Vector of residuals.
    - `coefs::Vector{Fl}`: Vector of coefficients.
    - `components_indexes::Dict{String, Vector{Int}}`: Dictionary containing indexes for different components.

    # Returns
    - `Dict`: Dictionary containing variances for each innovation component.

"""
function get_variances(
    model::StructuralModel,
    ε::Vector{Fl},
    coefs::Vector{Fl},
    components_indexes::Dict{String,Vector{Int}},
)::Dict where {Fl}
    model_innovations = get_model_innovations(model)

    variances = Dict()
    for component in model_innovations
        variances[component] = var(coefs[components_indexes[component]])
    end
    variances["ε"] = var(ε)
    return variances
end

"""
    get_model_innovations(model::StructuralModel)::Vector

    Returns a vector containing the model innovations based on the input parameters.

    # Arguments
    - `model::StructuralModel`: StructuralModel object.

    # Returns
    - `Vector`: Vector containing the model innovations.

"""
function get_model_innovations(model::StructuralModel)
    model_innovations = String[]
    if model.stochastic_level
        push!(model_innovations, "ξ")
    end

    if model.stochastic_trend
        push!(model_innovations, "ζ")
    end

    if model.stochastic_seasonal
        push!(model_innovations, "ω")
    end
    return model_innovations
end

"""
    get_innovation_functions(model::StructuralModel, innovation::String)::Function

    Returns the innovation function based on the input innovation string.

    # Arguments
    - `model::StructuralModel`: StructuralModel object.
    - `innovation::String`: Innovation string.
    - steps_ahead::Int: Number of steps ahead.

    # Returns

"""
function get_innovation_simulation_X(
    model::StructuralModel, innovation::String, steps_ahead::Int
)
    if innovation == "ξ"
        return create_ξ(length(model.y) + steps_ahead + 1, 0)
    elseif innovation == "ζ"
        return create_ζ(length(model.y) + steps_ahead + 1, 0, 1)
    elseif innovation == "ω"
        return create_ω(length(model.y) + steps_ahead + 1, model.freq_seasonal, 0, 1)
    end
end

isfitted(model::StructuralModel) = isnothing(model.output) ? false : true

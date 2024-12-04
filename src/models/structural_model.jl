@doc raw"""
Instantiates a Structural State Space Learning model.

    StructuralModel(
        y::Union{Vector,Matrix};
        level::Bool=true,
        stochastic_level::Bool=true,
        trend::Bool=true,
        stochastic_trend::Bool=true,
        seasonal::Bool=true,
        stochastic_seasonal::Bool=true,
        freq_seasonal::Union{Int, Vector{Int}}=12,
        outlier::Bool=true,
        ζ_ω_threshold::Int=12,
        Exogenous_X::Matrix=if typeof(y) <: Vector
            zeros(length(y), 0)
        else
            zeros(size(y, 1), 0)
        end,
    )

A Structural State Space Learning model that can have level, stochastic_level, trend, stochastic_trend, seasonal, stochastic_seasonal, outlier and Exogenous components. Each component should be specified by Booleans.

These models take the general form:

```math
\begin{gather*}
    \begin{aligned}
    y_1 &= \mu_1 + \gamma_1 + X_1\beta + \sum_{\tau=1}^1D_{\tau,1} \,o_\tau + \varepsilon_1\\
    y_2 &= \mu_1 + \xi_2 + \nu_1 + \gamma_2 + X_2\beta + \sum_{\tau=1}^2D_{\tau,2} \,o_\tau + \varepsilon_2 \\
    y_t &= \mu_1 + \sum_{\tau=2}^{t}\xi_\tau  + (t-1)\nu_1 + \sum_{\tau=2}^{t-1}(t-\tau)\zeta_\tau + \gamma_{m_t} + X_t\beta + \sum_{\tau=1}^tD_{\tau,t} \,o_\tau + \varepsilon_t, \quad \forall t = \{3, \ldots, s\} \label{cor_t2}\\
    y_t &= \mu_1  + \sum_{\tau=2}^{t}\xi_\tau + (t-1)\nu_1 + \sum_{\tau=2}^{t-1}(t-\tau)\zeta_\tau + \gamma_{m_{t}} + \sum_{\tau \in M_{t}}(\omega_{{\tau}} -  \omega_{\tau-1})+  X_t\beta + \sum_{\tau=1}^tD_{\tau,t} \,o_\tau + \varepsilon_t, \quad \forall t=s+1,\ldots,T \\
 %   \zeta_t, \xi_t, \omega_t =0, \; \forall \; t > T
    \end{aligned}
\end{gather*}
```

The notation is as follows: ``y_t`` represents the observation vector at time ``t``, ``\mu_1`` denotes the initial level component (intercept), and ``\xi_t`` refers to the stochastic level component. Similarly, ``\nu_1`` corresponds to the deterministic slope component, and ``\zeta_t`` represents the stochastic slope component. The seasonal effects are described by ``\gamma_{m_t}`` for the deterministic seasonal component and ``\omega_{\tau}`` for the stochastic seasonal component.

The exogenous component is represented by ``X``, with ``\beta`` as the associated coefficients. Outlier effects are captured by the dummy outlier matrix ``D`` and its corresponding coefficients ``o``. Finally, ``\varepsilon_t`` denotes the irregular term.

# References
 * Ramos, André, & Valladão, Davi, & Street, Alexandre.
   Time Series Analysis by State Space Learning

# Example
```julia
y = rand(100)
model = StructuralModel(y)
```
"""
mutable struct StructuralModel <: StateSpaceLearningModel
    y::Union{Vector,Matrix}
    X::Matrix
    level::Bool
    stochastic_level::Bool
    trend::Bool
    stochastic_trend::Bool
    seasonal::Bool
    stochastic_seasonal::Bool
    freq_seasonal::Union{Int, Vector{Int}}
    outlier::Bool
    ζ_ω_threshold::Int
    n_exogenous::Int
    output::Union{Vector{Output},Output,Nothing}

    function StructuralModel(
        y::Union{Vector,Matrix};
        level::Bool=true,
        stochastic_level::Bool=true,
        trend::Bool=true,
        stochastic_trend::Bool=true,
        seasonal::Bool=true,
        stochastic_seasonal::Bool=true,
        freq_seasonal::Union{Int, Vector{Int}}=12,
        outlier::Bool=true,
        ζ_ω_threshold::Int=12,
        Exogenous_X::Matrix=if typeof(y) <: Vector
            zeros(length(y), 0)
        else
            zeros(size(y, 1), 0)
        end,
    )
        n_exogenous = size(Exogenous_X, 2)
        @assert !has_intercept(Exogenous_X) "Exogenous matrix must not have an intercept column"
        if typeof(y) <: Vector
            @assert seasonal ? length(y) > minimum(freq_seasonal) : true "Time series must be longer than the seasonal period"
        else
            @assert seasonal ? size(y, 1) > minimum(freq_seasonal) : true "Time series must be longer than the seasonal period"
        end
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
    ξ_matrix = Matrix{AbstractFloat}(undef, T + steps_ahead, T - 1)
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
    ζ_matrix = Matrix{AbstractFloat}(undef, T + steps_ahead, T - 2)
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
create_ω(T::Int, freq_seasonal::Int, steps_ahead::Int, ζ_ω_threshold::Int)::Matrix

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
    create_initial_states_Matrix(
    T::Int, freq_seasonal::Union{Int, Vector{Int}}, steps_ahead::Int, level::Bool, trend::Bool, seasonal::Bool
)::Matrix

    Creates an initial states matrix based on the input parameters.

    # Arguments
    - `T::Int`: Length of the original time series.
    - `freq_seasonal::Union{Int, Vector{Int}}`: Seasonal period.
    - `steps_ahead::Int`: Number of steps ahead.
    - `level::Bool`: Flag for considering level component.
    - `trend::Bool`: Flag for considering trend component.
    - `seasonal::Bool`: Flag for considering seasonal component.

    # Returns
    - `Matrix`: Initial states matrix constructed based on the input parameters.

"""
function create_initial_states_Matrix(
    T::Int, freq_seasonal::Union{Int, Vector{Int}}, steps_ahead::Int, level::Bool, trend::Bool, seasonal::Bool
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
        for s in freq_seasonal
            γ1_matrix = zeros(T + steps_ahead, s)
            for t in 1:(T + steps_ahead)
                γ1_matrix[t, t % s == 0 ? s : t % s] = 1.0
            end
            initial_states_matrix = hcat(initial_states_matrix, γ1_matrix)
        end
    end

    return initial_states_matrix
end

"""
create_X(
    level::Bool,
    stochastic_level::Bool,
    trend::Bool,
    stochastic_trend::Bool,
    seasonal::Bool,
    stochastic_seasonal::Bool,
    freq_seasonal::Union{Int, Vector{Int}},
    outlier::Bool,
    ζ_ω_threshold::Int,
    Exogenous_X::Matrix{Fl},
    steps_ahead::Int=0,
    Exogenous_Forecast::Matrix{Tl}=zeros(steps_ahead, size(Exogenous_X, 2)),
) where {Fl <: AbstractFloat, Tl <: AbstractFloat}

    Creates the StateSpaceLearning matrix X based on the model type and input parameters.

    # Arguments
    - `level::Bool`: Flag for considering level component.
    - `stochastic_level::Bool`: Flag for considering stochastic level component.
    - `trend::Bool`: Flag for considering trend component.
    - `stochastic_trend::Bool`: Flag for considering stochastic trend component.
    - `seasonal::Bool`: Flag for considering seasonal component.
    - `stochastic_seasonal::Bool`: Flag for considering stochastic seasonal component.
    - `freq_seasonal::Union{Int, Vector{Int}}`: Seasonal period.
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
    freq_seasonal::Union{Int, Vector{Int}},
    outlier::Bool,
    ζ_ω_threshold::Int,
    Exogenous_X::Matrix{Fl},
    steps_ahead::Int=0,
    Exogenous_Forecast::Matrix{Tl}=zeros(steps_ahead, size(Exogenous_X, 2)),
) where {Fl<:AbstractFloat,Tl<:AbstractFloat}
    T = size(Exogenous_X, 1)

    ξ_matrix = stochastic_level ? create_ξ(T, steps_ahead) : zeros(T + steps_ahead, 0)
    ζ_matrix = if stochastic_trend
        create_ζ(T, steps_ahead, ζ_ω_threshold)
    else
        zeros(T + steps_ahead, 0)
    end

    ω_matrix = zeros(T + steps_ahead, 0)
    if stochastic_seasonal
        for s in freq_seasonal
            ω_matrix = hcat(ω_matrix, create_ω(T, s, steps_ahead, ζ_ω_threshold))
        end
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
function get_components_indexes(model::StructuralModel)::Dict

    Generates indexes dict for different components based on the model type and input parameters.

    # Arguments
    - `model::StructuralModel`: StructuralModel object.

    # Returns
    - `Dict`: Dictionary containing the corresponding indexes for each component of the model.

"""
function get_components_indexes(model::StructuralModel)::Dict
    T = typeof(model.y) <: Vector ? length(model.y) : size(model.y, 1)

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

    γ_indexes = Vector{Int}[]
    if model.seasonal
        for s in model.freq_seasonal
            γ_s_indexes = collect((FINAL_INDEX + 1):(FINAL_INDEX + s))
            initial_states_indexes = vcat(initial_states_indexes, γ_s_indexes)
            FINAL_INDEX += length(γ_s_indexes)
            push!(γ_indexes, γ_s_indexes)
        end
    else
        push!(γ_indexes, Int[])
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

    ω_indexes = Vector{Int}[]
    if model.stochastic_seasonal
        for s in model.freq_seasonal
            ω_s_indexes = collect(
                (FINAL_INDEX + 1):(FINAL_INDEX + ω_size(
                    T, s, model.ζ_ω_threshold
                )),
            )
            FINAL_INDEX += length(ω_s_indexes)
            push!(ω_indexes, ω_s_indexes)
        end
    else
        push!(ω_indexes, Int[])
    end

    if model.outlier
        o_indexes = collect((FINAL_INDEX + 1):(FINAL_INDEX + o_size(T)))
        FINAL_INDEX += length(o_indexes)
    else
        o_indexes = Int[]
    end

    exogenous_indexes = collect((FINAL_INDEX + 1):(FINAL_INDEX + model.n_exogenous))

    components_indexes_dict = Dict(
        "μ1" => μ1_indexes,
        "ν1" => ν1_indexes,
        "ξ" => ξ_indexes,
        "ζ" => ζ_indexes,
        "o" => o_indexes,
        "Exogenous_X" => exogenous_indexes,
        "initial_states" => initial_states_indexes,
    )

    for (i, s) in enumerate(model.freq_seasonal)
        components_indexes_dict["γ1_$s"] = γ_indexes[i]
        components_indexes_dict["ω_$s"] = ω_indexes[i]
    end

    return components_indexes_dict
end

"""
get_variances(
    model::StructuralModel,
    ε::Vector{Fl},
    coefs::Vector{Tl},
    components_indexes::Dict{String,Vector{Int}},
)::Dict where {Fl, Tl}

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
    coefs::Vector{Tl},
    components_indexes::Dict{String,Vector{Int}},
)::Dict where {Fl,Tl}
    model_innovations = get_model_innovations(model)

    variances = Dict()
    for component in model_innovations
        variances[component] = var(coefs[components_indexes[component]])
    end
    variances["ε"] = var(ε)
    return variances
end

"""
get_variances(
    model::StructuralModel,
    ε::Vector{Vector{Fl}},
    coefs::Vector{Vector{Tl}},
    components_indexes::Dict{String,Vector{Int}},
)::Vector{Dict} where {Fl, Tl}

    Calculates variances for each innovation component and for the residuals.

    # Arguments
    - `model::StructuralModel`: StructuralModel object.
    - `ε::Vector{Vector{Fl}}`: Vector of residuals.
    - `coefs::Vector{Vector{Fl}}`: Vector of coefficients.
    - `components_indexes::Dict{String, Vector{Int}}`: Dictionary containing indexes for different components.

    # Returns
    - `Vector{Dict}`: Dictionary containing variances for each innovation component.

"""
function get_variances(
    model::StructuralModel,
    ε::Vector{Vector{Fl}},
    coefs::Vector{Vector{Tl}},
    components_indexes::Dict{String,Vector{Int}},
)::Vector{Dict} where {Fl,Tl}
    model_innovations = get_model_innovations(model)

    variances_vec = Dict[]

    for i in eachindex(coefs)
        variances = Dict()
        for component in model_innovations
            variances[component] = var(coefs[i][components_indexes[component]])
        end
        variances["ε"] = var(ε[i])
        push!(variances_vec, variances)
    end
    return variances_vec
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
        for s in model.freq_seasonal
            push!(model_innovations, "ω_$s")
        end
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
    elseif occursin("ω_", innovation)
        s = parse(Int, split(innovation, "_")[2])
        return create_ω(length(model.y) + steps_ahead + 1, s, 0, 1)
    end
end

isfitted(model::StructuralModel) = isnothing(model.output) ? false : true

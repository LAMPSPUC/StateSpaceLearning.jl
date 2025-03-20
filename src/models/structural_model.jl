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
        stochastic_start::Int=1,
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
    freq_seasonal::Union{Int,Vector{Int}}
    cycle_period::Union{Union{Int,<:AbstractFloat},Vector{Int},Vector{<:AbstractFloat}}
    cycle_matrix::Vector{Matrix}
    stochastic_cycle::Bool
    outlier::Bool
    ζ_ω_threshold::Int
    stochastic_start::Int
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
        freq_seasonal::Union{Int,Vector{Int}}=12,
        cycle_period::Union{Union{Int,<:AbstractFloat},Vector{Int},Vector{<:AbstractFloat}}=0,
        dumping_cycle::Float64=1.0,
        stochastic_cycle::Bool=false,
        outlier::Bool=true,
        ζ_ω_threshold::Int=12,
        stochastic_start::Int=1,
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
        @assert 1 <= stochastic_start < length(y) "stochastic_start must be greater than or equal to 1"
        @assert 0 < dumping_cycle <= 1 "dumping_cycle must be greater than 0 and less than or equal to 1"
        if cycle_period != 0 && !isempty(cycle_period)
            if typeof(cycle_period) <: Vector
                cycle_matrix = Vector{Matrix}(undef, length(cycle_period))
                for i in eachindex(cycle_period)
                    A = dumping_cycle * cos(2 * pi / cycle_period[i])
                    B = dumping_cycle * sin(2 * pi / cycle_period[i])
                    cycle_matrix[i] = [A B; -B A]
                end
            else
                cycle_matrix = Vector{Matrix}(undef, 1)
                A = dumping_cycle * cos(2 * pi / cycle_period)
                B = dumping_cycle * sin(2 * pi / cycle_period)
                cycle_matrix[1] = [A B; -B A]
            end
        else
            cycle_matrix = Vector{Matrix}(undef, 0)
        end

        if typeof(freq_seasonal) <: Vector
            @assert all(freq_seasonal .> 0) "Seasonal period must be greater than 0"
        end

        if typeof(cycle_period) <: Vector
            @assert all(cycle_period .>= 0) "Cycle period must be greater than or equal to 0"
        end

        if cycle_period == 0
            @assert !stochastic_cycle "stochastic_cycle must be false if cycle_period is 0"
        end

        X = create_X(
            level,
            stochastic_level,
            trend,
            stochastic_trend,
            seasonal,
            stochastic_seasonal,
            freq_seasonal,
            cycle_matrix,
            stochastic_cycle,
            outlier,
            ζ_ω_threshold,
            stochastic_start,
            Exogenous_X,
        )
        # convert y format into vector or matrix of AbstractFloat
        if typeof(y) <: Vector
            y = convert(Vector{AbstractFloat}, y)
        else
            y = convert(Matrix{AbstractFloat}, y)
        end
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
            cycle_period,
            cycle_matrix,
            stochastic_cycle,
            outlier,
            ζ_ω_threshold,
            stochastic_start,
            n_exogenous,
            nothing,
        )
    end
end

"""
    ξ_size(T::Int, stochastic_start::Int)::Int

    Calculates the size of ξ innovation matrix based on the input T.

    # Arguments
    - `T::Int`: Length of the original time series.
    - `stochastic_start::Int`: parameter to set at which time stamp the stochastic component starts.

    # Returns
    - `Int`: Size of ξ calculated from T.

"""
ξ_size(T::Int, stochastic_start::Int)::Int = T - max(2, stochastic_start)

"""
ζ_size(T::Int, ζ_ω_threshold::Int, stochastic_start::Int)::Int

    Calculates the size of ζ innovation matrix based on the input T.

    # Arguments
    - `T::Int`: Length of the original time series.
    - `ζ_ω_threshold::Int`: Stabilize parameter ζ.
    - `stochastic_start::Int`: parameter to set at which time stamp the stochastic component starts.

    # Returns
    - `Int`: Size of ζ calculated from T.

"""
ζ_size(T::Int, ζ_ω_threshold::Int, stochastic_start::Int)::Int =
    max(0, T - ζ_ω_threshold - max(2, stochastic_start))

"""
ω_size(T::Int, s::Int, stochastic_start::Int)::Int

    Calculates the size of ω innovation matrix based on the input T.

    # Arguments
    - `T::Int`: Length of the original time series.
    - `s::Int`: Seasonal period.
    - `stochastic_start::Int`: parameter to set at which time stamp the stochastic component starts.

    # Returns
    - `Int`: Size of ω calculated from T.

"""
ω_size(T::Int, s::Int, ζ_ω_threshold::Int, stochastic_start::Int)::Int =
    max(0, T - ζ_ω_threshold - s + 1 - max(0, max(2, stochastic_start) - s))

"""
o_size(T::Int, stochastic_start::Int)::Int

    Calculates the size of outlier matrix based on the input T.

    # Arguments
    - `T::Int`: Length of the original time series.
    - `stochastic_start::Int`: parameter to set at which time stamp the stochastic component starts.

    # Returns
    - `Int`: Size of o calculated from T.

"""
o_size(T::Int, stochastic_start::Int)::Int = T - max(1, stochastic_start) + 1

"""
    ϕ_size(T::Int, ζ_ω_threshold::Int, stochastic_start::Int)::Int

    Calculates the size of ϕ innovation matrix based on the input T.

    # Arguments
    - `T::Int`: Length of the original time series.
    - `ζ_ω_threshold::Int`: Stabilize parameter ζ.
    - `stochastic_start::Int`: parameter to set at which time stamp the stochastic component starts.

    # Returns
    - `Int`: Size of ϕ calculated from T.
"""
function ϕ_size(T::Int, ζ_ω_threshold::Int, stochastic_start::Int)
    ζ_ω_threshold = ζ_ω_threshold == 0 ? 1 : ζ_ω_threshold
    if stochastic_start == 1
        return (2 * (T - max(2, stochastic_start) + 1) - (ζ_ω_threshold * 2)) - 2
    else
        return (2 * (T - max(2, stochastic_start) + 1) - (ζ_ω_threshold * 2))
    end
end

"""
    create_ξ(T::Int, steps_ahead::Int, stochastic_start::Int)::Matrix

    Creates a matrix of innovations ξ based on the input sizes, and the desired steps ahead (this is necessary for the forecast function)

    # Arguments
    - `T::Int`: Length of the original time series.
    - `steps_ahead::Int`: Number of steps ahead (for estimation purposes this should be set at 0).
    - `stochastic_start::Int`: parameter to set at which time stamp the stochastic component starts.

    # Returns
    - `Matrix`: Matrix of innovations ξ constructed based on the input sizes.

"""
function create_ξ(T::Int, steps_ahead::Int, stochastic_start::Int)::Matrix
    stochastic_start = max(2, stochastic_start)
    ξ_matrix = zeros(T + steps_ahead, T - stochastic_start + 1)
    ones_indexes = findall(
        I -> Tuple(I)[1] - (stochastic_start - 2) > Tuple(I)[2],
        CartesianIndices((T + steps_ahead, T - stochastic_start)),
    )
    ξ_matrix[ones_indexes] .= 1
    return ξ_matrix[:, 1:(end - 1)]
end

"""
create_ζ(T::Int, steps_ahead::Int, ζ_ω_threshold::Int, stochastic_start::Int)::Matrix

    Creates a matrix of innovations ζ based on the input sizes, and the desired steps ahead (this is necessary for the forecast function).

    # Arguments
    - `T::Int`: Length of the original time series.
    - `steps_ahead::Int`: Number of steps ahead (for estimation purposes this should be set at 0).
    - `ζ_ω_threshold::Int`: Stabilize parameter ζ.
    - `stochastic_start::Int`: parameter to set at which time stamp the stochastic component starts.

    # Returns
    - `Matrix`: Matrix of innovations ζ constructed based on the input sizes.

"""
function create_ζ(
    T::Int, steps_ahead::Int, ζ_ω_threshold::Int, stochastic_start::Int
)::Matrix
    stochastic_start = max(2, stochastic_start)
    ζ_matrix = zeros(T + steps_ahead, T - stochastic_start)

    for t in 2:(T + steps_ahead)
        if t < T
            len = t - stochastic_start
            ζ_matrix[t, 1:len] .= len:-1:1
        else
            ζ_matrix[t, :] .= (t - stochastic_start):-1:(t - T + 1)
        end
    end
    return ζ_matrix[:, 1:(end - ζ_ω_threshold)]
end

"""
create_ω(T::Int, freq_seasonal::Int, steps_ahead::Int, ζ_ω_threshold::Int, stochastic_start::Int)::Matrix

    Creates a matrix of innovations ω based on the input sizes, and the desired steps ahead (this is necessary for the forecast function).

    # Arguments
    - `T::Int`: Length of the original time series.
    - `freq_seasonal::Int`: Seasonal period.
    - `steps_ahead::Int`: Number of steps ahead (for estimation purposes this should be set at 0).
    - `ζ_ω_threshold::Int`: Stabilize parameter ζ.
    
    # Returns
    - `Matrix`: Matrix of innovations ω constructed based on the input sizes.

"""
function create_ω(
    T::Int, freq_seasonal::Int, steps_ahead::Int, ζ_ω_threshold::Int, stochastic_start::Int
)::Matrix
    stochastic_start = max(2, stochastic_start)
    ω_matrix_size = T - freq_seasonal + 1
    stochastic_start_diff = max(0, stochastic_start - freq_seasonal)

    ω_matrix = zeros(T + steps_ahead, ω_matrix_size - stochastic_start_diff)
    for t in (freq_seasonal + 1):(T + steps_ahead)
        ωₜ_coefs = zeros(ω_matrix_size)
        Mₜ = Int(floor((t - 1) / freq_seasonal))
        lag₁ = [t - j * freq_seasonal for j in 0:(Mₜ - 1)]
        lag₂ = [t - j * freq_seasonal - 1 for j in 0:(Mₜ - 1)]
        ωₜ_coefs[lag₁[lag₁ .<= ω_matrix_size + (freq_seasonal - 1)] .- (freq_seasonal - 1)] .=
            1
        ωₜ_coefs[lag₂[0 .< lag₂ .<= ω_matrix_size + (freq_seasonal - 1)] .- (freq_seasonal - 1)] .=
            -1
        ω_matrix[t, :] = ωₜ_coefs[(1 + stochastic_start_diff):end]
    end
    return ω_matrix[:, 1:(end - ζ_ω_threshold)]
end

"""
create_o_matrix(T::Int, steps_ahead::Int, stochastic_start::Int)::Matrix

    Creates a matrix of outliers based on the input sizes, and the desired steps ahead (this is necessary for the forecast function).

    # Arguments
    - `T::Int`: Length of the original time series.
    - `steps_ahead::Int`: Number of steps ahead (for estimation purposes this should be set at 0).
    - `stochastic_start::Int`: parameter to set at which time stamp the stochastic component starts.
    
    # Returns
    - `Matrix`: Matrix of outliers constructed based on the input sizes.

"""
function create_o_matrix(T::Int, steps_ahead::Int, stochastic_start::Int)::Matrix
    stochastic_start = max(1, stochastic_start)
    rows = stochastic_start:T
    cols = 1:(T - stochastic_start + 1)
    values = ones(length(rows))
    o_matrix = sparse(rows, cols, values, T, T - stochastic_start + 1)

    return vcat(o_matrix, zeros(steps_ahead, length(cols)))
end

"""
create_ϕ(X_cycle::Matrix, T::Int, steps_ahead::Int, ζ_ω_threshold::Int, stochastic_start::Int)::Matrix

    Creates a matrix of innovations ϕ based on the input sizes, and the desired steps ahead (this is necessary for the forecast function).

    # Arguments
    - `X_cycle::Matrix`: deterministic Cycle matrix.
    - `T::Int`: Length of the original time series.
    - `steps_ahead::Int64`: Number of steps ahead (for estimation purposes this should be set at 0).
    - `ζ_ω_threshold::Int`: Stabilize parameter ζ.
    - `stochastic_start::Int`: parameter to set at which time stamp the stochastic component starts.

    # Returns
    - `Matrix`: Matrix of innovations ϕ constructed based on the input sizes.
"""
function create_ϕ(
    c_matrix::Matrix, T::Int, steps_ahead::Int, ζ_ω_threshold::Int, stochastic_start::Int
)::Matrix
    num_cols = 2 * (T - stochastic_start + 1)
    X = Matrix{Float64}(undef, T + steps_ahead, num_cols)

    for (idx, t) in enumerate(stochastic_start:T)
        X[:, 2 * (idx - 1) + 1] = vcat(
            zeros(t - 1), c_matrix[1:(T - t + 1 + steps_ahead), 1]
        )
        X[:, 2 * (idx - 1) + 2] = vcat(
            zeros(t - 1), c_matrix[1:(T - t + 1 + steps_ahead), 2]
        )
    end

    ζ_ω_threshold = ζ_ω_threshold == 0 ? 1 : ζ_ω_threshold 
    if stochastic_start == 1
        return X[:, 3:(end - (ζ_ω_threshold * 2))]
    else
        return X[:, 1:(end - (ζ_ω_threshold * 2))]
    end
end

"""
    create_deterministic_cycle_matrix(cycle_matrix::Vector{Matrix}, T::Int, steps_ahead::Int)::Vector{Matrix}

    Creates a deterministic cycle matrix based on the input parameters.

    # Arguments
    - `cycle_matrix::Vector{Matrix}`: Vector of cycle matrices.
    - `T::Int`: Length of the original time series.
    - `steps_ahead::Int`: Number of steps ahead.

    # Returns
    - `Vector{Matrix}`: Deterministic cycle matrix constructed based on the input parameters.
"""
function create_deterministic_cycle_matrix(
    cycle_matrix::Vector{Matrix}, T::Int, steps_ahead::Int
)::Vector{Matrix}
    deterministic_cycle_matrix = Vector{Matrix}(undef, length(cycle_matrix))
    for (idx, c_matrix) in enumerate(cycle_matrix)
        X_cycle = Matrix{Float64}(undef, T + steps_ahead, 2)
        cycle_matrix_term = c_matrix^0
        X_cycle[1, :] = cycle_matrix_term[1, :]
        for t in 2:(T + steps_ahead)
            cycle_matrix_term *= c_matrix
            X_cycle[t, :] = cycle_matrix_term[1, :]
        end
        deterministic_cycle_matrix[idx] = X_cycle
    end
    return deterministic_cycle_matrix
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
    T::Int,
    freq_seasonal::Union{Int,Vector{Int}},
    steps_ahead::Int,
    level::Bool,
    trend::Bool,
    seasonal::Bool,
    deterministic_cycle_matrix::Vector{Matrix},
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

    if !isempty(deterministic_cycle_matrix)
        for c_matrix in deterministic_cycle_matrix
            initial_states_matrix = hcat(initial_states_matrix, c_matrix)
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
    stochastic_start::Int
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
    - `stochastic_start::Int`: parameter to set at which time stamp the stochastic component starts.
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
    freq_seasonal::Union{Int,Vector{Int}},
    cycle_matrix::Vector{Matrix},
    stochastic_cycle::Bool,
    outlier::Bool,
    ζ_ω_threshold::Int,
    stochastic_start::Int,
    Exogenous_X::Matrix{Fl},
    steps_ahead::Int=0,
    Exogenous_Forecast::Matrix{Tl}=zeros(steps_ahead, size(Exogenous_X, 2)),
) where {Fl<:AbstractFloat,Tl<:AbstractFloat}
    T = size(Exogenous_X, 1)

    ξ_matrix = if stochastic_level
        create_ξ(T, steps_ahead, stochastic_start)
    else
        zeros(T + steps_ahead, 0)
    end
    ζ_matrix = if stochastic_trend
        create_ζ(T, steps_ahead, ζ_ω_threshold, stochastic_start)
    else
        zeros(T + steps_ahead, 0)
    end

    ω_matrix = zeros(T + steps_ahead, 0)
    if stochastic_seasonal
        for s in freq_seasonal
            ω_matrix = hcat(
                ω_matrix, create_ω(T, s, steps_ahead, ζ_ω_threshold, stochastic_start)
            )
        end
    end

    deterministic_cycle_matrix = create_deterministic_cycle_matrix(
        cycle_matrix, T, steps_ahead
    )
    ϕ_matrix = zeros(T + steps_ahead, 0)
    if stochastic_cycle
        for c_matrix in deterministic_cycle_matrix
            ϕ_matrix = hcat(
                ϕ_matrix,
                create_ϕ(c_matrix, T, steps_ahead, ζ_ω_threshold, stochastic_start),
            )
        end
    end

    o_matrix = if outlier
        create_o_matrix(T, steps_ahead, stochastic_start)
    else
        zeros(T + steps_ahead, 0)
    end

    initial_states_matrix = create_initial_states_Matrix(
        T, freq_seasonal, steps_ahead, level, trend, seasonal, deterministic_cycle_matrix
    )
    return hcat(
        initial_states_matrix,
        ξ_matrix,
        ζ_matrix,
        ω_matrix,
        ϕ_matrix,
        o_matrix,
        vcat(Exogenous_X, Exogenous_Forecast),
    )
end

"""
create_X(
    model::StructuralModel,
    Exogenous_X::Matrix{Fl},
    steps_ahead::Int=0,
    Exogenous_Forecast::Matrix{Tl}=zeros(steps_ahead, size(Exogenous_X, 2)),
) where {Fl <: AbstractFloat, Tl <: AbstractFloat}

    Creates the StateSpaceLearning matrix X based on the model and input parameters.

    # Arguments
    - `model::StructuralModel`: StructuralModel object.
    - `Exogenous_X::Matrix{Fl}`: Exogenous variables matrix.
    - `steps_ahead::Int`: Number of steps ahead (default: 0).
    - `Exogenous_Forecast::Matrix{Fl}`: Exogenous variables forecast matrix (default: zeros).

    # Returns
    - `Matrix`: StateSpaceLearning matrix X constructed based on the input parameters.
"""
function create_X(
    model::StructuralModel,
    Exogenous_X::Matrix{Fl},
    steps_ahead::Int=0,
    Exogenous_Forecast::Matrix{Tl}=zeros(steps_ahead, size(Exogenous_X, 2)),
) where {Fl<:AbstractFloat,Tl<:AbstractFloat}
    return create_X(
        model.level,
        model.stochastic_level,
        model.trend,
        model.stochastic_trend,
        model.seasonal,
        model.stochastic_seasonal,
        model.freq_seasonal,
        model.cycle_matrix,
        model.stochastic_cycle,
        model.outlier,
        model.ζ_ω_threshold,
        model.stochastic_start,
        Exogenous_X,
        steps_ahead,
        Exogenous_Forecast,
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
        ν1_indexes = [FINAL_INDEX + 1]
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
    end

    c_indexes = Vector{Int}[]
    if !isempty(model.cycle_matrix)
        for _ in eachindex(model.cycle_matrix)
            c_i_indexes = collect((FINAL_INDEX + 1):(FINAL_INDEX + 2))
            initial_states_indexes = vcat(initial_states_indexes, c_i_indexes)
            FINAL_INDEX += length(c_i_indexes)
            push!(c_indexes, c_i_indexes)
        end
    end

    if model.stochastic_level
        ξ_indexes = collect(
            (FINAL_INDEX + 1):(FINAL_INDEX + ξ_size(T, model.stochastic_start))
        )
        FINAL_INDEX += length(ξ_indexes)
    else
        ξ_indexes = Int[]
    end

    if model.stochastic_trend
        ζ_indexes = collect(
            (FINAL_INDEX + 1):(FINAL_INDEX + ζ_size(
                T, model.ζ_ω_threshold, model.stochastic_start
            )),
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
                    T, s, model.ζ_ω_threshold, model.stochastic_start
                )),
            )
            FINAL_INDEX += length(ω_s_indexes)
            push!(ω_indexes, ω_s_indexes)
        end
    else
        push!(ω_indexes, Int[])
    end

    ϕ_indexes = Vector{Int}[]
    if model.stochastic_cycle
        for _ in eachindex(model.cycle_matrix)
            ϕ_i_indexes = collect(
                (FINAL_INDEX + 1):(FINAL_INDEX + ϕ_size(
                    T, model.ζ_ω_threshold, model.stochastic_start
                )),
            )
            FINAL_INDEX += length(ϕ_i_indexes)
            push!(ϕ_indexes, ϕ_i_indexes)
        end
    else
        push!(ϕ_indexes, Int[])
    end

    if model.outlier
        o_indexes = collect(
            (FINAL_INDEX + 1):(FINAL_INDEX + o_size(T, model.stochastic_start))
        )
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
        if model.seasonal
            components_indexes_dict["γ1_$s"] = γ_indexes[i]
        end

        if model.stochastic_seasonal
            components_indexes_dict["ω_$s"] = ω_indexes[i]
        end
    end

    if !isempty(model.cycle_matrix)
        for i in eachindex(model.cycle_period)
            components_indexes_dict["c1_$i"] = c_indexes[i]
            if model.stochastic_cycle
                components_indexes_dict["ϕ_$i"] = ϕ_indexes[i]
            end
        end
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

    if model.stochastic_cycle
        for i in eachindex(model.cycle_period)
            push!(model_innovations, "ϕ_$i")
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
        return create_ξ(length(model.y) + steps_ahead + 1, 0, model.stochastic_start)
    elseif innovation == "ζ"
        return create_ζ(length(model.y) + steps_ahead + 1, 0, 1, model.stochastic_start)
    elseif occursin("ω_", innovation)
        s = parse(Int, split(innovation, "_")[2])
        return create_ω(length(model.y) + steps_ahead + 1, s, 0, 1, model.stochastic_start)
    elseif occursin("ϕ_", innovation)
        i = parse(Int, split(innovation, "_")[2])
        deterministic_cycle_matrix = create_deterministic_cycle_matrix(
            model.cycle_matrix, length(model.y), steps_ahead + 1
        )
        return create_ϕ(
            deterministic_cycle_matrix[i],
            length(model.y) + steps_ahead + 1,
            0,
            model.ζ_ω_threshold,
            model.stochastic_start,
        )
    end
end

isfitted(model::StructuralModel) = isnothing(model.output) ? false : true

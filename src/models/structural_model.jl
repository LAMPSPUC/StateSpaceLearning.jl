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
        cycle_period::Union{Union{Int,<:AbstractFloat},Vector{Int},Vector{<:AbstractFloat}}=0,
        stochastic_cycle::Bool=false,
        outlier::Bool=true,
        ζ_threshold::Int=12,
        ω_threshold::Int=12,
        ϕ_threshold::Int=12,
        stochastic_start::Int=1,
        exog::Matrix=zeros(length(y), 0),
        dynamic_exog_coefs::Union{Vector{<:Tuple}, Nothing}=nothing
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
    slope::Bool
    stochastic_slope::Bool
    seasonal::Bool
    stochastic_seasonal::Bool
    cycle::Bool
    stochastic_cycle::Bool
    freq_seasonal::Union{Int,Vector{Int}}
    cycle_period::Union{Union{Int,<:AbstractFloat},Vector{Int},Vector{<:AbstractFloat}}
    outlier::Bool
    ξ_threshold::Int
    ζ_threshold::Int
    ω_threshold::Int
    ϕ_threshold::Int
    stochastic_start::Int
    n_exogenous::Int
    dynamic_exog_coefs::Union{Vector{<:Tuple},Nothing}
    output::Union{Vector{Output},Output,Nothing}

    function StructuralModel(
        y::Union{Vector,Matrix};
        level::String="stochastic",
        slope::String="stochastic",
        seasonal::String="stochastic",
        cycle::String="none",
        freq_seasonal::Union{Int,Vector{Int}}=12,
        cycle_period::Union{Union{Int,<:AbstractFloat},Vector{Int},Vector{<:AbstractFloat}}=0,
        outlier::Bool=true,
        ξ_threshold::Int=1,
        ζ_threshold::Int=12,
        ω_threshold::Int=12,
        ϕ_threshold::Int=12,
        stochastic_start::Int=1,
        exog::Matrix=zeros(length(y), 0),
        dynamic_exog_coefs::Union{Vector{<:Tuple},Nothing}=nothing,
    )
        n_exogenous = size(exog, 2)

        @assert !has_intercept(exog) "Exogenous matrix must not have an intercept column"
        @assert 1 <= stochastic_start < length(y) "stochastic_start must be greater than or equal to 1 and smaller than the length of the time series"
        @assert level in ["deterministic", "stochastic", "none"] "level must be either deterministic, stochastic or no"
        @assert slope in ["deterministic", "stochastic", "none"] "slope must be either deterministic, stochastic or no"
        @assert seasonal in ["deterministic", "stochastic", "none"] "seasonal must be either deterministic, stochastic or no"
        @assert cycle in ["deterministic", "stochastic", "none"] "cycle must be either deterministic, stochastic or no"
        @assert seasonal != "none" ? length(y) > minimum(freq_seasonal) : true "Time series must be longer than the seasonal period if seasonal is added"

        if typeof(freq_seasonal) <: Vector
            (@assert all(freq_seasonal .> 0) "Seasonal period must be greater than 0")
        else
            (@assert freq_seasonal > 0 "Seasonal period must be greater than 0")
        end

        if typeof(cycle_period) <: Vector
            (@assert all(cycle_period .>= 0) "Cycle period must be greater than or equal to 0")
        else
            (@assert cycle_period >= 0 "Cycle period must be greater than or equal to 0")
        end

        if cycle_period == 0
            (@assert cycle == "none" "stochastic_cycle and cycle must be false if cycle_period is 0")
        else
            nothing
        end

        if !isnothing(dynamic_exog_coefs)
            @assert all(
                typeof(dynamic_exog_coefs[i][1]) <: Vector for
                i in eachindex(dynamic_exog_coefs)
            ) "The first element of each tuple in dynamic_exog_coefs must be a vector"
            @assert all(
                typeof(dynamic_exog_coefs[i][2]) <: String for
                i in eachindex(dynamic_exog_coefs)
            ) "The second element of each tuple in dynamic_exog_coefs must be a string"
            @assert all([
                length(dynamic_exog_coefs[i][1]) .== length(y) for
                i in eachindex(dynamic_exog_coefs)
            ]) "The exogenous features that will be combined with state space components must have the same length as the time series"
            @assert all(
                dynamic_exog_coefs[i][2] in ["level", "slope", "seasonal", "cycle"] for
                i in eachindex(dynamic_exog_coefs)
            ) "The second element of each tuple in dynamic_exog_coefs must be a string that is either level, slope, seasonal or cycle"
            for i in eachindex(dynamic_exog_coefs)
                if dynamic_exog_coefs[i][2] == "seasonal" ||
                    dynamic_exog_coefs[i][2] == "cycle"
                    @assert length(dynamic_exog_coefs[i]) == 3 "The tuple in dynamic_exog_coefs must have 3 elements if the second element is seasonal or cycle"
                    @assert typeof(dynamic_exog_coefs[i][3]) <: Int "The third element of each tuple in dynamic_exog_coefs must be an integer if the second element is seasonal or cycle"
                    @assert dynamic_exog_coefs[i][3] > 1 "The third element of each tuple in dynamic_exog_coefs must be greater than 1 if the second element is seasonal or cycle"
                end
            end
        end

        X = create_X(
            level in ["stochastic", "deterministic"],
            level == "stochastic",
            slope in ["stochastic", "deterministic"],
            slope == "stochastic",
            seasonal in ["stochastic", "deterministic"],
            seasonal == "stochastic",
            cycle in ["stochastic", "deterministic"],
            cycle == "stochastic",
            freq_seasonal,
            cycle_period,
            outlier,
            ξ_threshold,
            ζ_threshold,
            ω_threshold,
            ϕ_threshold,
            stochastic_start,
            exog,
            dynamic_exog_coefs,
        )
        # convert y format into vector of AbstractFloat
        y = convert(Vector{AbstractFloat}, y)

        return new(
            y,
            X,
            level in ["stochastic", "deterministic"],
            level == "stochastic",
            slope in ["stochastic", "deterministic"],
            slope == "stochastic",
            seasonal in ["stochastic", "deterministic"],
            seasonal == "stochastic",
            cycle in ["stochastic", "deterministic"],
            cycle == "stochastic",
            freq_seasonal,
            cycle_period,
            outlier,
            ξ_threshold,
            ζ_threshold,
            ω_threshold,
            ϕ_threshold,
            stochastic_start,
            n_exogenous,
            dynamic_exog_coefs,
            nothing,
        )
    end
end

"""
Create a Basic Structural Model.

# Arguments
- `y::Union{Vector,Matrix}`: Time series to be modeled.
- `level::String="stochastic"`: Level component of the model.
- `slope::String="stochastic"`: Slope component of the model.
- `seasonal::String="stochastic"`: Seasonal component of the model.
- `freq_seasonal::Union{Int,Vector{Int}}=12`: Seasonal period of the model.

# Returns
- `StructuralModel`: Basic Structural Model.
"""
function BasicStructuralModel(
    y::Union{Vector,Matrix};
    level::String="stochastic",
    slope::String="stochastic",
    seasonal::String="stochastic",
    freq_seasonal::Union{Int,Vector{Int}}=12,
)
    return StructuralModel(
        y;
        level=level,
        slope=slope,
        seasonal=seasonal,
        cycle="none",
        freq_seasonal=freq_seasonal,
        cycle_period=0,
        outlier=false,
        ξ_threshold=0,
        ζ_threshold=0,
        ω_threshold=0,
        ϕ_threshold=0,
        stochastic_start=1,
        exog=zeros(length(y), 0),
        dynamic_exog_coefs=nothing,
    )
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
ξ_size(T::Int, ξ_threshold::Int, stochastic_start::Int)::Int =
    T - max(2, stochastic_start) + 1 - ξ_threshold

"""
ζ_size(T::Int, ζ_threshold::Int, stochastic_start::Int)::Int

    Calculates the size of ζ innovation matrix based on the input T.

    # Arguments
    - `T::Int`: Length of the original time series.
    - `ζ_threshold::Int`: Stabilize parameter ζ.
    - `stochastic_start::Int`: parameter to set at which time stamp the stochastic component starts.

    # Returns
    - `Int`: Size of ζ calculated from T.

"""
ζ_size(T::Int, ζ_threshold::Int, stochastic_start::Int)::Int = max(
    0, T - ζ_threshold - max(2, stochastic_start)
)

"""
ω_size(T::Int, s::Int, ω_threshold::Int, stochastic_start::Int)::Int

    Calculates the size of ω innovation matrix based on the input T.

    # Arguments
    - `T::Int`: Length of the original time series.
    - `s::Int`: Seasonal period.
    - `ω_threshold::Int`: Stabilize parameter ω.
    - `stochastic_start::Int`: parameter to set at which time stamp the stochastic component starts.

    # Returns
    - `Int`: Size of ω calculated from T.

"""
ω_size(T::Int, s::Int, ω_threshold::Int, stochastic_start::Int)::Int = max(
    0, T - ω_threshold - s + 1 - max(0, max(2, stochastic_start) - s)
)

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
    ϕ_size(T::Int, ϕ_threshold::Int, stochastic_start::Int)::Int

    Calculates the size of ϕ innovation matrix based on the input T.

    # Arguments
    - `T::Int`: Length of the original time series.
    - `ϕ_threshold::Int`: Stabilize parameter ϕ.
    - `stochastic_start::Int`: parameter to set at which time stamp the stochastic component starts.

    # Returns
    - `Int`: Size of ϕ calculated from T.
"""
ϕ_size(T::Int, ϕ_threshold::Int, stochastic_start::Int)::Int = (
    2 * (T - max(2, stochastic_start) + 1) - (max(1, ϕ_threshold) * 2)
)

"""
    create_ξ(T::Int, ξ_threshold::Int, stochastic_start::Int)::Matrix

    Creates a matrix of innovations ξ based on the input sizes, and the desired steps ahead (this is necessary for the forecast function)

    # Arguments
    - `T::Int`: Length of the original time series.
    - `ξ_threshold::Int`: Stabilize parameter ξ.
    - `stochastic_start::Int`: parameter to set at which time stamp the stochastic component starts.

    # Returns
    - `Matrix`: Matrix of innovations ξ constructed based on the input sizes.

"""
function create_ξ(T::Int, ξ_threshold::Int, stochastic_start::Int)::Matrix
    stochastic_start = max(2, stochastic_start)
    ξ_matrix = zeros(T, T - stochastic_start + 1)
    ones_indexes = findall(
        I -> Tuple(I)[1] - (stochastic_start - 2) > Tuple(I)[2],
        CartesianIndices((T, T - stochastic_start + 1)),
    )
    ξ_matrix[ones_indexes] .= 1
    return ξ_matrix[:, 1:(end - ξ_threshold)]
end

"""
create_ζ(T::Int, ζ_threshold::Int, stochastic_start::Int)::Matrix

    Creates a matrix of innovations ζ based on the input sizes, and the desired steps ahead (this is necessary for the forecast function).

    # Arguments
    - `T::Int`: Length of the original time series.
    - `ζ_threshold::Int`: Stabilize parameter ζ.
    - `stochastic_start::Int`: parameter to set at which time stamp the stochastic component starts.

    # Returns
    - `Matrix`: Matrix of innovations ζ constructed based on the input sizes.

"""
function create_ζ(T::Int, ζ_threshold::Int, stochastic_start::Int)::Matrix
    stochastic_start = max(2, stochastic_start)
    ζ_matrix = zeros(T, T - stochastic_start)

    for t in 2:T
        if t < T
            len = t - stochastic_start
            ζ_matrix[t, 1:len] .= len:-1:1
        else
            ζ_matrix[t, :] .= (t - stochastic_start):-1:(t - T + 1)
        end
    end
    return ζ_matrix[:, 1:(end - ζ_threshold)]
end

"""
create_ω(T::Int, freq_seasonal::Int, ω_threshold::Int, stochastic_start::Int)::Matrix

    Creates a matrix of innovations ω based on the input sizes, and the desired steps ahead (this is necessary for the forecast function).

    # Arguments
    - `T::Int`: Length of the original time series.
    - `freq_seasonal::Int`: Seasonal period.
    - `ω_threshold::Int`: Stabilize parameter ω.
    
    # Returns
    - `Matrix`: Matrix of innovations ω constructed based on the input sizes.

"""
function create_ω(
    T::Int, freq_seasonal::Int, ω_threshold::Int, stochastic_start::Int
)::Matrix
    stochastic_start = max(2, stochastic_start)
    ω_matrix_size = max(0, T - freq_seasonal + 1)
    stochastic_start_diff = max(0, stochastic_start - freq_seasonal)

    ω_matrix = zeros(T, ω_matrix_size - stochastic_start_diff)
    for t in (freq_seasonal + 1):T
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
    return ω_matrix[:, 1:(end - ω_threshold)]
end

"""
create_o_matrix(T::Int, stochastic_start::Int)::Matrix

    Creates a matrix of outliers based on the input sizes, and the desired steps ahead (this is necessary for the forecast function).

    # Arguments
    - `T::Int`: Length of the original time series.
    - `stochastic_start::Int`: parameter to set at which time stamp the stochastic component starts.
    
    # Returns
    - `Matrix`: Matrix of outliers constructed based on the input sizes.

"""
function create_o_matrix(T::Int, stochastic_start::Int)::Matrix
    stochastic_start = max(1, stochastic_start)
    rows = stochastic_start:T
    cols = 1:(T - stochastic_start + 1)
    values = ones(length(rows))
    o_matrix = sparse(rows, cols, values, T, T - stochastic_start + 1)

    return o_matrix
end

"""
create_ϕ(c_period::Union{Int, Fl}, T::Int, ϕ_threshold::Int, stochastic_start::Int)::Matrix

    Creates a matrix of innovations ϕ based on the input sizes, and the desired steps ahead (this is necessary for the forecast function).

    # Arguments
    - `c_period::Union{Int, Fl}`: Cycle period.
    - `T::Int`: Length of the original time series.
    - `ϕ_threshold::Int`: Stabilize parameter ϕ.
    - `stochastic_start::Int`: parameter to set at which time stamp the stochastic component starts.

    # Returns
    - `Matrix`: Matrix of innovations ϕ constructed based on the input sizes.
"""
function create_ϕ(
    c_period::Union{Int,Fl}, T::Int, ϕ_threshold::Int, stochastic_start::Int
)::Matrix where {Fl<:AbstractFloat}
    X = Matrix{Float64}(undef, T, 0)
    λ = 2 * pi * (1:T) / c_period

    for t in max(2, stochastic_start):(T - max(1, ϕ_threshold)) # one of last two columns might be full of zeros
        X_t = hcat(cos.(λ), sin.(λ))
        X_t[1:(t - 1), :] .= 0
        X = hcat(X, X_t)
    end

    return round.(X; digits=5)
end

"""
create_deterministic_seasonal(T::Int, s::Int)::Matrix

    Creates a matrix of deterministic seasonal components based on the input sizes.

    # Arguments
    - `T::Int`: Length of the original time series.
    - `s::Int`: Seasonal period.
"""
function create_deterministic_seasonal(T::Int, s::Int)::Matrix
    γ1_matrix = zeros(T, s)
    for t in 1:T
        γ1_matrix[t, t % s == 0 ? s : t % s] = 1.0
    end
    return γ1_matrix
end

"""
create_deterministic_cycle(T::Int, c_period::Union{Int, Fl})::Matrix where {Fl<:AbstractFloat}

    Creates a matrix of deterministic cycle components based on the input sizes.

    # Arguments
    - `T::Int`: Length of the original time series.
    - `c_period::Int`: Cycle period.
"""
function create_deterministic_cycle(
    T::Int, c_period::Union{Int,Fl}
)::Matrix where {Fl<:AbstractFloat}
    λ = 2 * pi * (1:T) / c_period
    cycle1_matrix = hcat(cos.(λ), sin.(λ))
    return round.(cycle1_matrix; digits=5)
end

"""
    create_initial_states_Matrix(
    T::Int, freq_seasonal::Union{Int, Vector{Int}}, level::Bool, trend::Bool, seasonal::Bool, cycle::Bool, cycle_period::Union{Int,Vector{Int}}
)::Matrix

    Creates an initial states matrix based on the input parameters.

    # Arguments
    - `T::Int`: Length of the original time series.
    - `freq_seasonal::Union{Int, Vector{Int}}`: Seasonal period.
    - `level::Bool`: Flag for considering level component.
    - `trend::Bool`: Flag for considering trend component.
    - `seasonal::Bool`: Flag for considering seasonal component.
    - `cycle::Bool`: Flag for considering cycle component.
    - `cycle_period::Union{Int,Vector{Int}}`: Cycle period.

    # Returns
    - `Matrix`: Initial states matrix constructed based on the input parameters.

"""
function create_initial_states_Matrix(
    T::Int,
    freq_seasonal::Union{Int,Vector{Int}},
    level::Bool,
    trend::Bool,
    seasonal::Bool,
    cycle::Bool,
    cycle_period::Union{Union{Int,<:AbstractFloat},Vector{Int},Vector{<:AbstractFloat}},
)::Matrix
    initial_states_matrix = zeros(T, 0)
    if level
        initial_states_matrix = hcat(initial_states_matrix, ones(T, 1))
    else
        nothing
    end
    if trend
        initial_states_matrix = hcat(initial_states_matrix, vcat([0], collect(1:(T - 1))))
    else
        nothing
    end

    if seasonal
        for s in freq_seasonal
            γ1_matrix = create_deterministic_seasonal(T, s)
            initial_states_matrix = hcat(initial_states_matrix, γ1_matrix)
        end
    end

    if cycle
        for c_period in cycle_period
            cycle1_matrix = create_deterministic_cycle(T, c_period)
            initial_states_matrix = hcat(initial_states_matrix, cycle1_matrix)
        end
    end

    return initial_states_matrix
end

"""
create_dynamic_exog_coefs_matrix(dynamic_exog_coefs::Vector{<:Tuple}, T::Int,ζ_threshold::Int, ω_threshold::Int, ϕ_threshold::Int, stochastic_start::Int)::Matrix

    Creates a matrix of combination components based on the input parameters.

    # Arguments
    - `dynamic_exog_coefs::Vector{<:Tuple}`: Vector of tuples containing the combination components.
    - `T::Int`: Length of the original time series.
    - `ζ_threshold::Int`: Stabilize parameter ζ.
    - `ω_threshold::Int`: Stabilize parameter ω.
    - `ϕ_threshold::Int`: Stabilize parameter ϕ.
    - `stochastic_start::Int`: parameter to set at which time stamp the stochastic component starts.

    # Returns
    - `Matrix`: Matrix of combination components constructed based on the input parameters.
"""
function create_dynamic_exog_coefs_matrix(
    dynamic_exog_coefs::Vector{<:Tuple},
    T::Int,
    ξ_threshold::Int,
    ζ_threshold::Int,
    ω_threshold::Int,
    ϕ_threshold::Int,
    stochastic_start::Int,
)::Matrix
    state_components_dict = Dict{String,Matrix}()
    dynamic_exog_coefs_matrix = zeros(T, 0)
    for combination in dynamic_exog_coefs
        if combination[2] == "level"
            if haskey(state_components_dict, "level")
                nothing
            else
                state_components_dict["level"] = hcat(
                    ones(T, 1), create_ξ(T, ξ_threshold, stochastic_start)
                )
            end
            key_name = "level"
        elseif combination[2] == "slope"
            if haskey(state_components_dict, "slope")
                nothing
            else
                state_components_dict["slope"] = hcat(
                    vcat([0], collect(1:(T - 1))),
                    create_ζ(T, ζ_threshold, stochastic_start),
                )
            end
            key_name = "slope"
        elseif combination[2] == "seasonal"
            if haskey(state_components_dict, "seasonal_$(combination[3])")
                nothing
            else
                state_components_dict["seasonal_$(combination[3])"] = hcat(
                    create_deterministic_seasonal(T, combination[3]),
                    create_ω(T, combination[3], ω_threshold, stochastic_start),
                )
            end
            key_name = "seasonal_$(combination[3])"
        elseif combination[2] == "cycle"
            if haskey(state_components_dict, "cycle_$(combination[3])")
                nothing
            else
                state_components_dict["cycle_$(combination[3])"] = hcat(
                    create_deterministic_cycle(T, combination[3]),
                    create_ϕ(combination[3], T, ϕ_threshold, stochastic_start),
                )
            end
            key_name = "cycle_$(combination[3])"
        end
        dynamic_exog_coefs_matrix = hcat(
            dynamic_exog_coefs_matrix, combination[1] .* state_components_dict[key_name]
        )
    end
    return dynamic_exog_coefs_matrix
end

"""
create_forecast_dynamic_exog_coefs_matrix(dynamic_exog_coefs::Vector{<:Tuple}, T::Int, steps_ahead::Int, ζ_threshold::Int, ω_threshold::Int, ϕ_threshold::Int, stochastic_start::Int)::Matrix

    Creates a matrix of combination components based on the input parameters.

    # Arguments
    - `dynamic_exog_coefs::Vector{<:Tuple}`: Vector of tuples containing the combination components.
    - `T::Int`: Length of the original time series.
    - `steps_ahead::Int`: Steps ahead.
    - `ξ_threshold::Int`: Stabilize parameter ξ.
    - `ζ_threshold::Int`: Stabilize parameter ζ.
    - `ω_threshold::Int`: Stabilize parameter ω.
    - `ϕ_threshold::Int`: Stabilize parameter ϕ.
    - `stochastic_start::Int`: parameter to set at which time stamp the stochastic component starts.

    # Returns
    - `Matrix`: Matrix of combination components constructed based on the input parameters.
"""
function create_forecast_dynamic_exog_coefs_matrix(
    dynamic_exog_coefs::Vector{<:Tuple},
    T::Int,
    steps_ahead::Int,
    ξ_threshold::Int,
    ζ_threshold::Int,
    ω_threshold::Int,
    ϕ_threshold::Int,
    stochastic_start::Int,
)::Matrix
    state_components_dict = Dict{String,Matrix}()
    dynamic_exog_coefs_matrix = zeros(steps_ahead, 0)
    for combination in dynamic_exog_coefs
        if combination[2] == "level"
            if haskey(state_components_dict, "level")
                nothing
            else
                state_components_dict["level"] = hcat(
                    ones(T + steps_ahead, 1),
                    create_ξ(T + steps_ahead, ξ_threshold, stochastic_start),
                )[
                    (end - steps_ahead + 1):end, 1:combination[4]
                ]
            end
            key_name = "level"
        elseif combination[2] == "slope"
            if haskey(state_components_dict, "slope")
                nothing
            else
                state_components_dict["slope"] = hcat(
                    vcat([0], collect(1:(T + steps_ahead - 1))),
                    create_ζ(T + steps_ahead, ζ_threshold, stochastic_start),
                )[
                    (end - steps_ahead + 1):end, 1:combination[4]
                ]
            end
            key_name = "slope"
        elseif combination[2] == "seasonal"
            if haskey(state_components_dict, "seasonal_$(combination[3])")
                nothing
            else
                state_components_dict["seasonal_$(combination[3])"] = hcat(
                    create_deterministic_seasonal(T + steps_ahead, combination[3]),
                    create_ω(
                        T + steps_ahead, combination[3], ω_threshold, stochastic_start
                    ),
                )[
                    (end - steps_ahead + 1):end, 1:combination[4]
                ]
            end
            key_name = "seasonal_$(combination[3])"
        elseif combination[2] == "cycle"
            if haskey(state_components_dict, "cycle_$(combination[3])")
                nothing
            else
                state_components_dict["cycle_$(combination[3])"] = hcat(
                    create_deterministic_cycle(T + steps_ahead, combination[3]),
                    create_ϕ(
                        combination[3], T + steps_ahead, ϕ_threshold, stochastic_start
                    ),
                )[
                    (end - steps_ahead + 1):end, 1:combination[4]
                ]
            end
            key_name = "cycle_$(combination[3])"
        end
        dynamic_exog_coefs_matrix = hcat(
            dynamic_exog_coefs_matrix, combination[1] .* state_components_dict[key_name]
        )
    end
    return dynamic_exog_coefs_matrix
end

"""
create_X(
    level::Bool,
    stochastic_level::Bool,
    slope::Bool,
    stochastic_slope::Bool,
    seasonal::Bool,
    stochastic_seasonal::Bool,
    cycle::Bool,
    stochastic_cycle::Bool,
    freq_seasonal::Union{Int,Vector{Int}},
    cycle_period::Union{Int,Vector{Int}},
    outlier::Bool,
    ξ_threshold::Int,
    ζ_threshold::Int,
    ω_threshold::Int,
    ϕ_threshold::Int,
    stochastic_start::Int,
    exog::Matrix{Fl},
) where {Fl<:AbstractFloat}

    Creates the StateSpaceLearning matrix X based on the model type and input parameters.

    # Arguments
    - `level::Bool`: Flag for considering level component.
    - `stochastic_level::Bool`: Flag for considering stochastic level component.
    - `slope::Bool`: Flag for considering slope component.
    - `stochastic_slope::Bool`: Flag for considering stochastic slope component.
    - `seasonal::Bool`: Flag for considering seasonal component.
    - `stochastic_seasonal::Bool`: Flag for considering stochastic seasonal component.
    - `cycle::Bool`: Flag for considering cycle component.
    - `stochastic_cycle::Bool`: Flag for considering stochastic cycle component.
    - `freq_seasonal::Union{Int, Vector{Int}}`: Seasonal period.
    - `cycle_period::Union{Int,Vector{Int}}`: Cycle period.
    - `outlier::Bool`: Flag for considering outlier component.
    - `ξ_threshold::Int`: Stabilize parameter ξ.
    - `ζ_threshold::Int`: Stabilize parameter ζ.
    - `ω_threshold::Int`: Stabilize parameter ω.
    - `ϕ_threshold::Int`: Stabilize parameter ϕ.
    - `stochastic_start::Int`: parameter to set at which time stamp the stochastic component starts.
    - `exog::Matrix{Fl}`: Exogenous variables matrix.

    # Returns
    - `Matrix`: StateSpaceLearning matrix X constructed based on the input parameters.
"""
function create_X(
    level::Bool,
    stochastic_level::Bool,
    slope::Bool,
    stochastic_slope::Bool,
    seasonal::Bool,
    stochastic_seasonal::Bool,
    cycle::Bool,
    stochastic_cycle::Bool,
    freq_seasonal::Union{Int,Vector{Int}},
    cycle_period::Union{Union{Int,<:AbstractFloat},Vector{Int},Vector{<:AbstractFloat}},
    outlier::Bool,
    ξ_threshold::Int,
    ζ_threshold::Int,
    ω_threshold::Int,
    ϕ_threshold::Int,
    stochastic_start::Int,
    exog::Matrix{Fl},
    dynamic_exog_coefs::Union{Vector{<:Tuple},Nothing},
) where {Fl<:AbstractFloat}
    T = size(exog, 1)

    ξ_matrix = if stochastic_level
        create_ξ(T, ξ_threshold, stochastic_start)
    else
        zeros(T, 0)
    end
    ζ_matrix = if stochastic_slope
        create_ζ(T, ζ_threshold, stochastic_start)
    else
        zeros(T, 0)
    end

    ω_matrix = zeros(T, 0)
    if stochastic_seasonal
        for s in freq_seasonal
            ω_matrix = hcat(ω_matrix, create_ω(T, s, ω_threshold, stochastic_start))
        end
    end

    ϕ_matrix = zeros(T, 0)
    if stochastic_cycle
        for c_period in cycle_period
            ϕ_matrix = hcat(ϕ_matrix, create_ϕ(c_period, T, ϕ_threshold, stochastic_start))
        end
    end

    o_matrix = if outlier
        create_o_matrix(T, stochastic_start)
    else
        zeros(T, 0)
    end

    initial_states_matrix = create_initial_states_Matrix(
        T, freq_seasonal, level, slope, seasonal, cycle, cycle_period
    )

    dynamic_exog_coefs_matrix = if !isnothing(dynamic_exog_coefs)
        create_dynamic_exog_coefs_matrix(
            dynamic_exog_coefs,
            T,
            ξ_threshold,
            ζ_threshold,
            ω_threshold,
            ϕ_threshold,
            stochastic_start,
        )
    else
        zeros(T, 0)
    end

    return hcat(
        initial_states_matrix,
        ξ_matrix,
        ζ_matrix,
        ω_matrix,
        ϕ_matrix,
        o_matrix,
        exog,
        dynamic_exog_coefs_matrix,
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

    if model.slope
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
    if model.cycle
        for _ in eachindex(model.cycle_period)
            c_i_indexes = collect((FINAL_INDEX + 1):(FINAL_INDEX + 2))
            initial_states_indexes = vcat(initial_states_indexes, c_i_indexes)
            FINAL_INDEX += length(c_i_indexes)
            push!(c_indexes, c_i_indexes)
        end
    end

    if model.stochastic_level
        ξ_indexes = collect(
            (FINAL_INDEX + 1):(FINAL_INDEX + ξ_size(
                T, model.ξ_threshold, model.stochastic_start
            )),
        )
        FINAL_INDEX += length(ξ_indexes)
    else
        ξ_indexes = Int[]
    end

    if model.stochastic_slope
        ζ_indexes = collect(
            (FINAL_INDEX + 1):(FINAL_INDEX + ζ_size(
                T, model.ζ_threshold, model.stochastic_start
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
                    T, s, model.ω_threshold, model.stochastic_start
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
        for _ in eachindex(model.cycle_period)
            ϕ_i_indexes = collect(
                (FINAL_INDEX + 1):(FINAL_INDEX + ϕ_size(
                    T, model.ϕ_threshold, model.stochastic_start
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
    FINAL_INDEX += length(exogenous_indexes)

    dynamic_exog_coefs_indexes = collect((FINAL_INDEX + 1):size(model.X, 2))

    components_indexes_dict = Dict(
        "μ1" => μ1_indexes,
        "ν1" => ν1_indexes,
        "ξ" => ξ_indexes,
        "ζ" => ζ_indexes,
        "o" => o_indexes,
        "exog" => exogenous_indexes,
        "initial_states" => initial_states_indexes,
        "dynamic_exog_coefs" => dynamic_exog_coefs_indexes,
    )

    for (i, s) in enumerate(model.freq_seasonal)
        if model.seasonal
            components_indexes_dict["γ1_$s"] = γ_indexes[i]
        end

        if model.stochastic_seasonal
            components_indexes_dict["ω_$s"] = ω_indexes[i]
        end
    end

    if model.cycle
        for i in eachindex(model.cycle_period)
            components_indexes_dict["c1_$(model.cycle_period[i])"] = c_indexes[i]
            if model.stochastic_cycle
                components_indexes_dict["ϕ_$(model.cycle_period[i])"] = ϕ_indexes[i]
            end
        end
    end

    if !isnothing(model.dynamic_exog_coefs)
        for i in eachindex(model.dynamic_exog_coefs)
            if model.dynamic_exog_coefs[i][2] == "level"
                components_indexes_dict["dynamic_exog_coefs_$(i)"] = collect(
                    (FINAL_INDEX + 1):(FINAL_INDEX + 1 + ξ_size(
                        T, model.ξ_threshold, model.stochastic_start
                    )),
                )
                FINAL_INDEX += length(components_indexes_dict["dynamic_exog_coefs_$(i)"])
            elseif model.dynamic_exog_coefs[i][2] == "slope"
                components_indexes_dict["dynamic_exog_coefs_$(i)"] = collect(
                    (FINAL_INDEX + 1):(FINAL_INDEX + 1 + ζ_size(
                        T, model.ζ_threshold, model.stochastic_start
                    )),
                )
                FINAL_INDEX += length(components_indexes_dict["dynamic_exog_coefs_$(i)"])
            elseif model.dynamic_exog_coefs[i][2] == "seasonal"
                components_indexes_dict["dynamic_exog_coefs_$(i)"] = collect(
                    (FINAL_INDEX + 1):(FINAL_INDEX + model.dynamic_exog_coefs[i][3] + ω_size(
                        T,
                        model.dynamic_exog_coefs[i][3],
                        model.ω_threshold,
                        model.stochastic_start,
                    )),
                )
                FINAL_INDEX += length(components_indexes_dict["dynamic_exog_coefs_$(i)"])
            elseif model.dynamic_exog_coefs[i][2] == "cycle"
                components_indexes_dict["dynamic_exog_coefs_$(i)"] = collect(
                    (FINAL_INDEX + 1):(FINAL_INDEX + 2 + ϕ_size(
                        T, model.ϕ_threshold, model.stochastic_start
                    )),
                )
                FINAL_INDEX += length(components_indexes_dict["dynamic_exog_coefs_$(i)"])
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

    if model.stochastic_slope
        push!(model_innovations, "ζ")
    end

    if model.stochastic_seasonal
        for s in model.freq_seasonal
            push!(model_innovations, "ω_$s")
        end
    end

    if model.stochastic_cycle
        for i in model.cycle_period
            push!(model_innovations, "ϕ_$i")
        end
    end

    if !isnothing(model.dynamic_exog_coefs)
        for i in eachindex(model.dynamic_exog_coefs)
            push!(model_innovations, "dynamic_exog_coefs_$i")
        end
    end
    return model_innovations
end

"""
    get_trend_decomposition(model::StructuralModel, components::Dict, slope::Vector{AbstractFloat})::Vector{AbstractFloat}

    Returns the level component and associated innovation vectors.

    # Arguments
    - `model::StructuralModel`: StructuralModel object.
    - `components::Dict` : Components dict.
    - `slope::Vector{AbstractFloat}`: Time-series of the slope component.

    # Returns
    - `Vector{AbstractFloat}`: Time-series of the level component.

"""
function get_trend_decomposition(
    model::StructuralModel, components::Dict, slope::Vector{AbstractFloat}
)::Vector{AbstractFloat}
    T = size(model.y, 1)
    trend = Vector{AbstractFloat}(undef, T)

    if model.level
        trend[1] = components["μ1"]["Coefs"][1]
    else
        trend[1] = 0.0
    end

    if model.stochastic_level && !isempty(components["ξ"]["Coefs"])
        ξ = vcat(
            zeros(max(2, model.stochastic_start) - 1),
            components["ξ"]["Coefs"],
            zeros(model.ξ_threshold),
        )
        @assert length(ξ) == T
    else
        ξ = zeros(AbstractFloat, T)
    end

    for t in 2:T
        trend[t] = trend[t - 1] + slope[t] + ξ[t]
    end

    return trend
end

"""
    get_slope_decomposition(model::StructuralModel, components::Dict)::Vector{AbstractFloat}

    Returns the slope component and associated innovation vectors.

    # Arguments
    - `model::StructuralModel`: StructuralModel object.
    - `components::Dict`: Components dict..

    # Returns
    - `Vector{AbstractFloat}`: Time-series of the slope component.

"""
function get_slope_decomposition(
    model::StructuralModel, components::Dict
)::Vector{AbstractFloat}
    T = size(model.y, 1)
    slope = Vector{AbstractFloat}(undef, T)

    if model.slope
        slope[1] = components["ν1"]["Coefs"][1]
    else
        slope[1] = 0.0
    end

    if model.stochastic_slope && !isempty(components["ζ"]["Coefs"])
        ζ = vcat(
            zeros(max(2, model.stochastic_start)),
            components["ζ"]["Coefs"],
            zeros(model.ζ_threshold),
        )
        @assert length(ζ) == T
    else
        ζ = zeros(AbstractFloat, T)
    end

    for t in 2:T
        slope[t] = slope[t - 1] + ζ[t]
    end

    return slope
end

"""
    get_seasonal_decomposition(model::StructuralModel, components::Dict, s::Int)::Vector{AbstractFloat}

    Returns the seasonality component and associated innovation vectors.

    # Arguments
    - `model::StructuralModel`: StructuralModel object.
    - `components::Dict`: Components dict.
    - `s::Int`: Seasonal frequency.	

    # Returns
    - `Vector{AbstractFloat}`: Time-series of the seasonality component.

"""
function get_seasonal_decomposition(
    model::StructuralModel, components::Dict, s::Int
)::Vector{AbstractFloat}
    T = size(model.y, 1)
    seasonal = Vector{AbstractFloat}(undef, T)

    if model.seasonal
        seasonal[1:s] = components["γ1_$(s)"]["Coefs"]
    else
        seasonal[1:s] = zeros(AbstractFloat, s)
    end

    if model.stochastic_seasonal && !isempty(components["ω_$(s)"]["Coefs"])
        ω = vcat(
            zeros(s - 1 + max(0, max(2, model.stochastic_start) - s)),
            components["ω_$(s)"]["Coefs"],
            zeros(model.ω_threshold),
        )
        @assert length(ω) == T
    else
        ω = zeros(AbstractFloat, T)
    end

    for t in (s + 1):T
        seasonal[t] = seasonal[t - s] + ω[t] - ω[t - 1]
    end

    return seasonal
end

"""
    get_cycle_decomposition(model::StructuralModel, components::Dict, cycle_period::Union{AbstractFloat, Int})::Tuple{Vector{AbstractFloat}, Vector{AbstractFloat}}

    Returns the cycle component and associated innovation vectors.

    # Arguments
    - `model::StructuralModel`: StructuralModel object.
    - `components::Dict`: Components dict.
    - `cycle_period::Union{AbstractFloat, Int}`: Cycle period.

    # Returns
    - `Tuple{Vector{AbstractFloat}, Vector{AbstractFloat}}`: Tuple containing the cycle component and the cycle component hat.

"""
function get_cycle_decomposition(
    model::StructuralModel, components::Dict, cycle_period::Union{AbstractFloat,Int}
)::Vector{AbstractFloat}
    T = size(model.y, 1)
    cycle = Vector{AbstractFloat}(undef, T)

    if cycle_period != 0
        λ = 2 * pi * (1:T) / cycle_period
        c1 = components["c1_$(cycle_period)"]["Coefs"]

        cycle[1] = (dot(c1, [cos(λ[1]), sin(λ[1])]))

        if model.stochastic_cycle
            ϕ_cos = vcat(
                zeros(max(2, model.stochastic_start) - 1),
                components["ϕ_$(cycle_period)"]["Coefs"][1:2:end],
                zeros(
                    min(max(1, model.ϕ_threshold), T - max(2, model.stochastic_start) + 1)
                ),
            )
            ϕ_sin = vcat(
                zeros(max(2, model.stochastic_start) - 1),
                components["ϕ_$(cycle_period)"]["Coefs"][2:2:end],
                zeros(
                    min(max(1, model.ϕ_threshold), T - max(2, model.stochastic_start) + 1)
                ),
            )
            @assert length(ϕ_cos) == T
            @assert length(ϕ_sin) == T
        else
            ϕ_cos = zeros(AbstractFloat, T)
            ϕ_sin = zeros(AbstractFloat, T)
        end

        for t in 2:T
            if max(2, model.stochastic_start) <= min(t, (T - max(1, model.ϕ_threshold)))
                ϕ_indexes =
                    max(2, model.stochastic_start):min(t, (T - max(1, model.ϕ_threshold)))
                cycle[t] =
                    dot(c1, [cos(λ[t]), sin(λ[t])]) + sum(
                        ϕ_cos[i] * cos(λ[t]) + ϕ_sin[i] * sin(λ[t]) for
                        i in eachindex(ϕ_indexes)
                    )
            else
                cycle[t] = dot(c1, [cos(λ[t]), sin(λ[t])])
            end
        end

    else
        cycle = zeros(AbstractFloat, T)
    end

    return cycle
end

"""
    get_model_decomposition(model::StructuralModel, components::Dict)::Dict  

    Returns a dictionary with the time series state and innovations for each component.

    # Arguments
    - `model::StructuralModel`: StructuralModel object.
    - `components::Dict`: Components dict.

    # Returns
    - `Dict`: Dictionary of time-series states and innovations.

"""
function get_model_decomposition(model::StructuralModel, components::Dict)::Dict
    freq_seasonal = model.freq_seasonal
    cycle_period = model.cycle_period
    model_decomposition = Dict()

    if model.slope
        slope = get_slope_decomposition(model, components)
        model_decomposition["slope"] = slope
    end

    if model.level || model.slope
        slope = model.slope ? slope : convert(Vector{AbstractFloat}, zeros(length(model.y)))
        trend = get_trend_decomposition(model, components, slope)
        model_decomposition["trend"] = trend
    end

    if model.seasonal
        for s in freq_seasonal
            seasonal = get_seasonal_decomposition(model, components, s)
            model_decomposition["seasonal_$s"] = seasonal
        end
    end

    if model.cycle
        for i in cycle_period
            cycle, cycle_hat = get_cycle_decomposition(model, components, i)
            model_decomposition["cycle_$i"] = cycle
            model_decomposition["cycle_hat_$i"] = cycle_hat
        end
    end
    return model_decomposition
end

"""
    simulate_states(
        model::StructuralModel, steps_ahead::Int, punctual::Bool, seasonal_innovation_simulation::Int
    )::Vector{AbstractFloat}

    Simulates the states of the model.

    # Arguments
    - `model::StructuralModel`: StructuralModel object.
    - `steps_ahead::Int`: Steps ahead.
    - `punctual::Bool`: Flag for considering punctual forecast.
    - `seasonal_innovation_simulation::Int`: Flag for considering seasonal innovation simulation.

    # Returns
    - `Vector{AbstractFloat}`: Vector of states.
"""
function simulate_states(
    model::StructuralModel,
    steps_ahead::Int,
    punctual::Bool,
    seasonal_innovation_simulation::Int,
    N_scenarios::Int,
)::Matrix{AbstractFloat}
    T = length(model.y)

    prediction = Matrix{AbstractFloat}(undef, steps_ahead, N_scenarios)

    if model.slope
        slope = deepcopy(model.output.decomposition["slope"])
        start_idx = max(2, model.stochastic_start) + 1
        final_idx = T - model.ζ_threshold
        if model.stochastic_slope && !punctual
            if seasonal_innovation_simulation != 0
                ζ_values = vcat(
                    zeros(start_idx - 1),
                    model.output.components["ζ"]["Coefs"],
                    zeros(model.ζ_threshold),
                )
            else
                if !isempty(model.output.components["ζ"]["Coefs"])
                    ζ_values = model.output.components["ζ"]["Coefs"]
                else
                    ζ_values = zeros(T)
                end
            end
        else
            ζ_values = zeros(T)
        end
        stochastic_slope_set = get_stochastic_values(
            ζ_values,
            steps_ahead,
            T,
            start_idx,
            final_idx,
            seasonal_innovation_simulation,
            N_scenarios,
        )
    else
        slope = zeros(T)
    end

    if model.level || model.slope
        trend = deepcopy(model.output.decomposition["trend"])
        start_idx = max(2, model.stochastic_start)
        final_idx = T - 1
        if model.stochastic_level && !punctual
            if seasonal_innovation_simulation != 0
                ξ_values = vcat(
                    zeros(start_idx - 1), model.output.components["ξ"]["Coefs"], zeros(1)
                )
            else
                if !isempty(model.output.components["ξ"]["Coefs"])
                    ξ_values = model.output.components["ξ"]["Coefs"]
                else
                    ξ_values = zeros(T)
                end
            end
        else
            ξ_values = zeros(T)
        end
        stochastic_level_set = get_stochastic_values(
            ξ_values,
            steps_ahead,
            T,
            start_idx,
            final_idx,
            seasonal_innovation_simulation,
            N_scenarios,
        )
    end

    if model.seasonal
        seasonals = [
            deepcopy(model.output.decomposition["seasonal_$s"]) for s in model.freq_seasonal
        ]
        start_idx = [
            model.freq_seasonal[i] - 1 +
            max(0, max(2, model.stochastic_start) - model.freq_seasonal[i]) for
            i in eachindex(model.freq_seasonal)
        ]
        final_idx = [T - model.ω_threshold for _ in eachindex(model.freq_seasonal)]
        if model.ω_threshold == 0
            final_ω = [
                model.output.components["ω_$(s)"]["Coefs"][end] for s in model.freq_seasonal
            ]
        else
            final_ω = [0.0 for _ in model.freq_seasonal]
        end
        if model.stochastic_seasonal && !punctual
            if seasonal_innovation_simulation != 0
                ω_values = [
                    vcat(
                        zeros(s - 1 + max(0, max(2, model.stochastic_start) - s)),
                        model.output.components["ω_$(s)"]["Coefs"],
                        zeros(model.ω_threshold),
                    ) for s in model.freq_seasonal
                ]
            else
                ω_values = []
                for s in model.freq_seasonal
                    if !isempty(model.output.components["ω_$(s)"]["Coefs"])
                        push!(ω_values, model.output.components["ω_$(s)"]["Coefs"])
                    else
                        push!(ω_values, zeros(T))
                    end
                end
            end
        else
            ω_values = [zeros(T) for _ in model.freq_seasonal]
        end
        stochastic_seasonals_set = [
            vcat(
                final_ω[i],
                get_stochastic_values(
                    ω_values[i],
                    steps_ahead,
                    T,
                    start_idx[i],
                    final_idx[i],
                    seasonal_innovation_simulation,
                    N_scenarios,
                ),
            ) for i in eachindex(model.freq_seasonal)
        ]
    end

    if model.cycle
        start_idx = [max(2, model.stochastic_start) for _ in eachindex(model.cycle_period)]
        final_idx = [T - max(1, model.ϕ_threshold) for _ in eachindex(model.cycle_period)]
        if model.stochastic_cycle && !punctual
            if seasonal_innovation_simulation != 0
                ϕ_cos_values = [
                    vcat(
                        zeros(max(2, model.stochastic_start) - 1),
                        model.output.components["ϕ_$(i)"]["Coefs"][1:2:end],
                        zeros(max(1, model.ϕ_threshold)),
                    ) for i in model.cycle_period
                ]
                ϕ_sin_values = [
                    vcat(
                        zeros(max(2, model.stochastic_start) - 1),
                        model.output.components["ϕ_$(i)"]["Coefs"][2:2:end],
                        zeros(max(1, model.ϕ_threshold)),
                    ) for i in model.cycle_period
                ]
            else
                ϕ_cos_values = []
                ϕ_sin_values = []
                for i in model.cycle_period
                    if !isempty(model.output.components["ϕ_$(i)"]["Coefs"])
                        push!(
                            ϕ_cos_values,
                            model.output.components["ϕ_$(i)"]["Coefs"][1:2:end],
                        )
                        push!(
                            ϕ_sin_values,
                            model.output.components["ϕ_$(i)"]["Coefs"][2:2:end],
                        )
                    else
                        push!(ϕ_cos_values, zeros(T))
                        push!(ϕ_sin_values, zeros(T))
                    end
                end
            end
        else
            ϕ_cos_values = [zeros(T) for _ in model.cycle_period]
            ϕ_sin_values = [zeros(T) for _ in model.cycle_period]
        end
        stochastic_cycles_cos_set = [
            get_stochastic_values(
                ϕ_cos_values[i],
                steps_ahead,
                T,
                start_idx[i],
                final_idx[i],
                seasonal_innovation_simulation,
                N_scenarios,
            ) for i in eachindex(model.cycle_period)
        ]
        stochastic_cycles_sin_set = [
            get_stochastic_values(
                ϕ_sin_values[i],
                steps_ahead,
                T,
                start_idx[i],
                final_idx[i],
                seasonal_innovation_simulation,
                N_scenarios,
            ) for i in eachindex(model.cycle_period)
        ]
    end

    if model.outlier && !punctual
        start_idx = 1
        final_idx = T
        outlier_values = model.output.components["o"]["Coefs"]
        stochastic_outliers_set = rand(outlier_values, steps_ahead, N_scenarios)
    end

    if !punctual
        stochastic_residuals_set = rand(
            model.output.ε[findall(i -> !isnan(i), model.output.ε)],
            steps_ahead,
            N_scenarios,
        )
    end

    model.slope ? slope = ones(T, N_scenarios) .* slope : nothing
    model.level ? trend = ones(T, N_scenarios) .* trend : nothing
    model.seasonal ? seasonals = [ones(T, N_scenarios) .* s for s in seasonals] : nothing
    for t in (T + 1):(T + steps_ahead)
        slope_t = if model.slope
            slope[end, :] + stochastic_slope_set[t - T, :]
        else
            zeros(N_scenarios)
        end

        trend_t = if (model.level || model.slope)
            trend[end, :] + slope[end, :] + stochastic_level_set[t - T, :]
        else
            zeros(N_scenarios)
        end

        if model.seasonal
            seasonals_t = []
            for i in eachindex(model.freq_seasonal)
                push!(
                    seasonals_t,
                    seasonals[i][t - model.freq_seasonal[i], :] +
                    stochastic_seasonals_set[i][t - T + 1, :] -
                    stochastic_seasonals_set[i][t - T, :],
                )
            end
        else
            seasonals_t = [zeros(N_scenarios) for _ in eachindex(model.freq_seasonal)]
        end

        if model.cycle_period != 0 && model.cycle_period != []
            cycles_t = []
            for i in eachindex(model.cycle_period)
                λ = 2 * pi * (1:(T + steps_ahead)) / model.cycle_period[i]

                if model.stochastic_cycle &&
                    !isempty(model.output.components["ϕ_$(model.cycle_period[i])"]["Coefs"])
                    ϕ_cos = model.output.components["ϕ_$(model.cycle_period[i])"]["Coefs"][1:2:end]
                    ϕ_sin = model.output.components["ϕ_$(model.cycle_period[i])"]["Coefs"][2:2:end]
                    cycle_t =
                        ones(N_scenarios) .* dot(
                            model.output.components["c1_$(model.cycle_period[i])"]["Coefs"],
                            [cos(λ[t]), sin(λ[t])],
                        ) +
                        ones(N_scenarios) .* sum(
                            ϕ_cos[j] * cos(λ[t]) + ϕ_sin[j] * sin(λ[t]) for
                            j in eachindex(ϕ_cos)
                        ) +
                        [
                            sum(
                                stochastic_cycles_cos_set[i][j] * cos(λ[t]) +
                                stochastic_cycles_sin_set[i][j] * sin(λ[t]) for
                                j in eachindex(stochastic_cycles_cos_set[i][1:(t - T), k])
                            ) for k in 1:N_scenarios
                        ]
                else
                    cycle_t =
                        ones(N_scenarios) .* dot(
                            model.output.components["c1_$(model.cycle_period[i])"]["Coefs"],
                            [cos(λ[t]), sin(λ[t])],
                        )
                end
                push!(cycles_t, cycle_t)
            end
        else
            cycles_t = [zeros(N_scenarios) for _ in eachindex(model.cycle_period)]
        end

        outlier_t = if (model.outlier && !punctual)
            stochastic_outliers_set[t - T, :]
        else
            zeros(N_scenarios)
        end
        residuals_t = !punctual ? stochastic_residuals_set[t - T, :] : zeros(N_scenarios)

        prediction[t - T, :] =
            trend_t + sum(seasonals_t) + sum(cycles_t) + outlier_t + residuals_t

        model.slope ? slope = vcat(slope, slope_t') : nothing
        model.level ? trend = vcat(trend, trend_t') : nothing
        if model.seasonal
            for i in eachindex(model.freq_seasonal)
                seasonals[i] = vcat(seasonals[i], seasonals_t[i]')
            end
        end
    end

    return prediction
end

"""
    forecast_dynamic_exog_coefs(model::StructuralModel, steps_ahead::Int, dynamic_exog_coefs_forecasts::Vector{<:Vector})::Vector{AbstractFloat}

    Returns the prediction of the combination components terms.

    # Arguments
    - `model::StructuralModel`: StructuralModel object.
    - `steps_ahead::Int`: Number of steps ahead for forecasting.
    - `dynamic_exog_coefs_forecasts::Vector{<:Vector}`: Vector of vectors of combination components forecasts.

    # Returns
    - `Vector{AbstractFloat}`: Vector of combination components forecasts.
"""
function forecast_dynamic_exog_coefs(
    model::StructuralModel, steps_ahead::Int, dynamic_exog_coefs_forecasts::Vector{<:Vector}
)::Vector{AbstractFloat}
    if !isempty(dynamic_exog_coefs_forecasts)
        T = length(model.y)
        dynamic_exog_coefs = Vector{Tuple}(undef, length(model.dynamic_exog_coefs))
        for i in eachindex(model.dynamic_exog_coefs)
            if model.dynamic_exog_coefs[i][2] == "level"
                n_coefs = 1 + ξ_size(T, model.ξ_threshold, model.stochastic_start)
                extra_param = ""
            elseif model.dynamic_exog_coefs[i][2] == "slope"
                n_coefs = 1 + ζ_size(T, model.ζ_threshold, model.stochastic_start)
                extra_param = ""
            elseif model.dynamic_exog_coefs[i][2] == "seasonal"
                n_coefs =
                    model.dynamic_exog_coefs[i][3] + ω_size(
                        T,
                        model.dynamic_exog_coefs[i][3],
                        model.ω_threshold,
                        model.stochastic_start,
                    )
                extra_param = model.dynamic_exog_coefs[i][3]
            elseif model.dynamic_exog_coefs[i][2] == "cycle"
                n_coefs = 2 + ϕ_size(T, model.ϕ_threshold, model.stochastic_start)
                extra_param = model.dynamic_exog_coefs[i][3]
            end
            dynamic_exog_coefs[i] = (
                dynamic_exog_coefs_forecasts[i],
                model.dynamic_exog_coefs[i][2],
                extra_param,
                n_coefs,
            )
        end
        dynamic_exog_coefs_forecasts_matrix = create_forecast_dynamic_exog_coefs_matrix(
            dynamic_exog_coefs,
            T,
            steps_ahead,
            model.ξ_threshold,
            model.ζ_threshold,
            model.ω_threshold,
            model.ϕ_threshold,
            model.stochastic_start,
        )
        dynamic_exog_coefs_prediction =
            dynamic_exog_coefs_forecasts_matrix *
            model.output.components["dynamic_exog_coefs"]["Coefs"]
    else
        dynamic_exog_coefs_prediction = zeros(steps_ahead)
    end
    return dynamic_exog_coefs_prediction
end

"""
    forecast(model::StructuralModel, steps_ahead::Int; Exogenous_Forecast::Matrix{Fl}=zeros(steps_ahead, 0), dynamic_exog_coefs_forecasts::Vector{<:Vector}=Vector{Vector}(undef, 0))::Vector{Dict}

    Returns a vector of dictionaries with the scenarios of each component, for each dependent time-series.

    # Arguments
    - `model::StructuralModel`: StructuralModel object.
    - `steps_ahead::Int`: Number of steps ahead for forecasting.
    - `Exogenous_Forecast::Matrix{Fl}`: Matrix of forecasts of exogenous variables.
    - `dynamic_exog_coefs_forecasts::Vector{<:Vector}`: Vector of vectors of combination components forecasts.

"""
function forecast(
    model::StructuralModel,
    steps_ahead::Int;
    Exogenous_Forecast::Matrix{Fl}=zeros(steps_ahead, 0),
    dynamic_exog_coefs_forecasts::Vector{<:Vector}=Vector{Vector}(undef, 0),
)::Vector{AbstractFloat} where {Fl<:AbstractFloat}
    states_prediction = simulate_states(model, steps_ahead, true, 0, 1)[:, 1]

    @assert size(Exogenous_Forecast, 1) == steps_ahead
    @assert all(
        length(dynamic_exog_coefs_forecasts[i]) == steps_ahead for
        i in eachindex(dynamic_exog_coefs_forecasts)
    )
    if !isnothing(model.dynamic_exog_coefs)
        (@assert length(dynamic_exog_coefs_forecasts) == length(model.dynamic_exog_coefs))
    else
        nothing
    end
    if (dynamic_exog_coefs_forecasts == Vector{Vector}(undef, 0))
        (@assert isnothing(model.dynamic_exog_coefs))
    else
        nothing
    end
    @assert size(Exogenous_Forecast, 2) == model.n_exogenous

    dynamic_exog_coefs_prediction = forecast_dynamic_exog_coefs(
        model, steps_ahead, dynamic_exog_coefs_forecasts
    )

    prediction =
        states_prediction +
        (Exogenous_Forecast * model.output.components["exog"]["Coefs"]) +
        dynamic_exog_coefs_prediction

    return prediction
end

"""
    simulate(model::StructuralModel, steps_ahead::Int, N_scenarios::Int; Exogenous_Forecast::Matrix{Fl}=zeros(steps_ahead, 0), dynamic_exog_coefs_forecasts::Vector{<:Vector}=Vector{Vector}(undef, 0), seasonal_innovation_simulation::Int=0, seed::Int=1234)::Matrix{AbstractFloat}

    Returns a matrix of scenarios of the states of the model.

    # Arguments
    - `model::StructuralModel`: StructuralModel object.
    - `steps_ahead::Int`: Number of steps ahead for forecasting.
    - `N_scenarios::Int`: Number of scenarios to simulate.
    - `Exogenous_Forecast::Matrix{Fl}`: Matrix of forecasts of exogenous variables.
    - `dynamic_exog_coefs_forecasts::Vector{<:Vector}`: Vector of vectors of combination components forecasts.
    - `seasonal_innovation_simulation::Int`: Number of seasonal innovation simulation.
    - `seed::Int`: Seed for the random number generator.

    # Returns
    - `Matrix{AbstractFloat}`: Matrix of scenarios of the states of the model.
"""
function simulate(
    model::StructuralModel,
    steps_ahead::Int,
    N_scenarios::Int;
    Exogenous_Forecast::Matrix{Fl}=zeros(steps_ahead, 0),
    dynamic_exog_coefs_forecasts::Vector{<:Vector}=Vector{Vector}(undef, 0),
    seasonal_innovation_simulation::Int=0,
    seed::Int=1234,
)::Matrix{AbstractFloat} where {Fl<:AbstractFloat}
    Random.seed!(seed)
    scenarios = simulate_states(
        model, steps_ahead, false, seasonal_innovation_simulation, N_scenarios
    )

    dynamic_exog_coefs_prediction = forecast_dynamic_exog_coefs(
        model, steps_ahead, dynamic_exog_coefs_forecasts
    )
    scenarios .+=
        (Exogenous_Forecast * model.output.components["exog"]["Coefs"]) +
        dynamic_exog_coefs_prediction

    return scenarios
end

isfitted(model::StructuralModel) = isnothing(model.output) ? false : true

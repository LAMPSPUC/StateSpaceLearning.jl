"""
    ξ_size(T::Int64)::Int64

    Calculates the size of ξ innovation matrix based on the input T.

    # Arguments
    - `T::Int64`: Length of the original time series.

    # Returns
    - `Int64`: Size of ξ calculated from T.

"""
ξ_size(T::Int64)::Int64 = T - 2

"""
ζ_size(T::Int64, ζ_ω_threshold::Int64)::Int64

    Calculates the size of ζ innovation matrix based on the input T.

    # Arguments
    - `T::Int64`: Length of the original time series.
    - `ζ_ω_threshold::Int64`: Stabilize parameter ζ.

    # Returns
    - `Int64`: Size of ζ calculated from T.

"""
ζ_size(T::Int64, ζ_ω_threshold::Int64)::Int64 = T-ζ_ω_threshold-2

"""
ω_size(T::Int64, s::Int64)::Int64

    Calculates the size of ω innovation matrix based on the input T.

    # Arguments
    - `T::Int64`: Length of the original time series.
    - `s::Int64`: Seasonal period.

    # Returns
    - `Int64`: Size of ω calculated from T.

"""
ω_size(T::Int64, s::Int64, ζ_ω_threshold::Int64)::Int64 = T - ζ_ω_threshold - s + 1

"""
    create_ξ(T::Int64, steps_ahead::Int64)::Matrix

    Creates a matrix of innovations ξ based on the input sizes, and the desired steps ahead (this is necessary for the forecast function)

    # Arguments
    - `T::Int64`: Length of the original time series.
    - `steps_ahead::Int64`: Number of steps ahead (for estimation purposes this should be set at 0).

    # Returns
    - `Matrix`: Matrix of innovations ξ constructed based on the input sizes.

"""
function create_ξ(T::Int64, steps_ahead::Int64)::Matrix
    ξ_matrix = Matrix{Float64}(undef, T+steps_ahead, T - 1)
    for t in 1:T+steps_ahead
        ξ_matrix[t, :] = t < T ? vcat(ones(t-1), zeros(T-t)) : ones(T-1)
    end
    
    return ξ_matrix[:, 1:end-1]
end

"""
create_ζ(T::Int64, steps_ahead::Int64, ζ_ω_threshold::Int64)::Matrix

    Creates a matrix of innovations ζ based on the input sizes, and the desired steps ahead (this is necessary for the forecast function).

    # Arguments
    - `T::Int64`: Length of the original time series.
    - `steps_ahead::Int64`: Number of steps ahead (for estimation purposes this should be set at 0).
    - `ζ_ω_threshold::Int64`: Stabilize parameter ζ.

    # Returns
    - `Matrix`: Matrix of innovations ζ constructed based on the input sizes.

"""
function create_ζ(T::Int64, steps_ahead::Int64, ζ_ω_threshold::Int64)::Matrix
    ζ_matrix = Matrix{Float64}(undef, T+steps_ahead, T - 2)
    for t in 1:T+steps_ahead
        ζ_matrix[t, :] = t < T ? vcat(collect(t-2:-1:1), zeros(T-2-length(collect(t-2:-1:1)))) : collect(t-2:-1:t-T+1)
    end
    return ζ_matrix[:, 1:end-ζ_ω_threshold]
end

"""
create_ω(T::Int64, s::Int64, steps_ahead::Int64)::Matrix

    Creates a matrix of innovations ω based on the input sizes, and the desired steps ahead (this is necessary for the forecast function).

    # Arguments
    - `T::Int64`: Length of the original time series.
    - `freq_seasonal::Int64`: Seasonal period.
    - `steps_ahead::Int64`: Number of steps ahead (for estimation purposes this should be set at 0).
    
    # Returns
    - `Matrix`: Matrix of innovations ω constructed based on the input sizes.

"""
function create_ω(T::Int64, freq_seasonal::Int64, steps_ahead::Int64, ζ_ω_threshold::Int64)::Matrix
    ω_matrix_size = ω_size(T, freq_seasonal, ζ_ω_threshold) + ζ_ω_threshold
    ω_matrix = zeros(T+steps_ahead, ω_matrix_size)
    for t in freq_seasonal+1:T+steps_ahead
        ωₜ_coefs = zeros(ω_matrix_size)
        Mₜ = Int64(floor((t-1)/freq_seasonal))
        lag₁ = [t - j*freq_seasonal for j in 0:Mₜ-1]
        lag₂ = [t - j*freq_seasonal - 1 for j in 0:Mₜ-1]
        ωₜ_coefs[lag₁[lag₁.<=ω_matrix_size+(freq_seasonal - 1)] .- (freq_seasonal - 1)] .= 1
        ωₜ_coefs[lag₂[0 .< lag₂ .<=ω_matrix_size+(freq_seasonal - 1)] .- (freq_seasonal - 1)] .= -1
        ω_matrix[t, :] = ωₜ_coefs
    end
    return ω_matrix[:, 1:end-ζ_ω_threshold]
end

"""
    create_initial_states_Matrix(T::Int64, s::Int64, steps_ahead::Int64, level::Bool, trend::Bool, seasonal::Bool)::Matrix

    Creates an initial states matrix based on the input parameters.

    # Arguments
    - `T::Int64`: Length of the original time series.
    - `freq_seasonal::Int64`: Seasonal period.
    - `steps_ahead::Int64`: Number of steps ahead.
    - `level::Bool`: Flag for considering level component.
    - `trend::Bool`: Flag for considering trend component.
    - `seasonal::Bool`: Flag for considering seasonal component.

    # Returns
    - `Matrix`: Initial states matrix constructed based on the input parameters.

"""
function create_initial_states_Matrix(T::Int64, freq_seasonal::Int64, steps_ahead::Int64, level::Bool, trend::Bool, seasonal::Bool)::Matrix

    initial_states_matrix = zeros(T+steps_ahead, 0)
    level ? initial_states_matrix = hcat(initial_states_matrix, ones(T+steps_ahead, 1)) : nothing
    trend ? initial_states_matrix = hcat(initial_states_matrix, vcat([0], collect(1:T+steps_ahead-1))) : nothing

    if seasonal
        γ1_matrix = zeros(T+steps_ahead, freq_seasonal)
        for t in 1:T+steps_ahead
            γ1_matrix[t, t % freq_seasonal == 0 ? freq_seasonal : t % freq_seasonal] = 1.0
        end
        return hcat(initial_states_matrix, γ1_matrix)
    end

    return initial_states_matrix

end

"""
create_X(model_input::Dict, Exogenous_X::Matrix{Fl}, steps_ahead::Int64=0, Exogenous_Forecast::Matrix{Fl}=zeros(steps_ahead, size(Exogenous_X, 2))) where Fl

    Creates the StateSpaceLearning matrix X based on the model type and input parameters.

    # Arguments
    - `model_type::String`: Type of model.
    - `Exogenous_X::Matrix{Fl}`: Exogenous variables matrix.
    - `steps_ahead::Int64`: Number of steps ahead (default: 0).
    - `Exogenous_Forecast::Matrix{Fl}`: Exogenous variables forecast matrix (default: zeros).

    # Returns
    - `Matrix`: StateSpaceLearning matrix X constructed based on the input parameters.
"""
function create_X(model_input::Dict, Exogenous_X::Matrix{Fl},
                  steps_ahead::Int64=0, Exogenous_Forecast::Matrix{Fl}=zeros(steps_ahead, size(Exogenous_X, 2))) where Fl

    outlier = model_input["outlier"]; ζ_ω_threshold = model_input["ζ_ω_threshold"]; T = size(Exogenous_X, 1)
    
    ξ_matrix = model_input["stochastic_level"] ? create_ξ(T, steps_ahead) : zeros(T+steps_ahead, 0)
    ζ_matrix = model_input["stochastic_trend"] ? create_ζ(T, steps_ahead, ζ_ω_threshold) : zeros(T+steps_ahead, 0)
    ω_matrix = model_input["stochastic_seasonal"] ? create_ω(T, model_input["freq_seasonal"], steps_ahead, ζ_ω_threshold) : zeros(T+steps_ahead, 0)
    o_matrix = outlier ? create_o_matrix(T, steps_ahead) : zeros(T+steps_ahead, 0)

    initial_states_matrix = create_initial_states_Matrix(T, model_input["freq_seasonal"], steps_ahead, model_input["level"], model_input["trend"], model_input["seasonal"])
    return hcat(initial_states_matrix, ξ_matrix, ζ_matrix, ω_matrix, o_matrix, vcat(Exogenous_X, Exogenous_Forecast))
    
end

"""
get_components_indexes(Exogenous_X::Matrix{Fl}, model_input::Dict)::Dict where Fl

    Generates indexes dict for different components based on the model type and input parameters.

    # Arguments
    - `Exogenous_X::Matrix{Fl}`: Exogenous variables matrix.
    - `model_input`: Dictionary containing the model input parameters.

    # Returns
    - `Dict`: Dictionary containing the corresponding indexes for each component of the model.

"""
function get_components_indexes(Exogenous_X::Matrix{Fl}, model_input::Dict)::Dict where Fl
    
    outlier = model_input["outlier"]; ζ_ω_threshold = model_input["ζ_ω_threshold"]; T = size(Exogenous_X, 1)

    FINAL_INDEX = 0

    if model_input["level"]
        μ1_indexes = [1]
        initial_states_indexes = [1]
        FINAL_INDEX += length(μ1_indexes)
    else
        μ1_indexes = Int64[]
        initial_states_indexes = Int64[]
    end

    if model_input["trend"]
        ν1_indexes = [2]
        initial_states_indexes = vcat(initial_states_indexes, ν1_indexes)
        FINAL_INDEX += length(ν1_indexes)
    else
        ν1_indexes = Int64[]
    end
    
    if model_input["seasonal"]
        γ1_indexes = collect(FINAL_INDEX+1:FINAL_INDEX+model_input["freq_seasonal"])
        initial_states_indexes = vcat(initial_states_indexes, γ1_indexes)
        FINAL_INDEX += length(γ1_indexes)
    else
        γ1_indexes = Int64[]
    end

    if model_input["stochastic_level"]
        ξ_indexes = collect(FINAL_INDEX+1:FINAL_INDEX+ξ_size(T))
        FINAL_INDEX += length(ξ_indexes)
    else
        ξ_indexes = Int64[]
    end

    if model_input["stochastic_trend"]
        ζ_indexes = collect(FINAL_INDEX+1:FINAL_INDEX+ζ_size(T, ζ_ω_threshold))
        FINAL_INDEX += length(ζ_indexes)
    else
        ζ_indexes = Int64[]
    end

    if model_input["stochastic_seasonal"]
        ω_indexes = collect(FINAL_INDEX+1:FINAL_INDEX+ω_size(T, model_input["freq_seasonal"], ζ_ω_threshold))
        FINAL_INDEX += length(ω_indexes)
    else
        ω_indexes = Int64[]
    end 

    if outlier
        o_indexes = collect(FINAL_INDEX+1:FINAL_INDEX+o_size(T))
        FINAL_INDEX += length(o_indexes)
    else
        o_indexes = Int64[]
    end

    exogenous_indexes = collect(FINAL_INDEX + 1:FINAL_INDEX + size(Exogenous_X, 2))

    return Dict("μ1" => μ1_indexes, "ν1" => ν1_indexes, "γ1" => γ1_indexes, 
                "ξ" => ξ_indexes, "ζ" => ζ_indexes, "ω" => ω_indexes, "o" => o_indexes, 
                "Exogenous_X" => exogenous_indexes, "initial_states" => initial_states_indexes)
end

"""
    get_variances(ε::Vector{Fl}, coefs::Vector{Fl}, components_indexes::Dict{String, Vector{Int64}})::Dict where Fl

    Calculates variances for each innovation component and for the residuals.

    # Arguments
    - `ε::Vector{Fl}`: Vector of residuals.
    - `coefs::Vector{Fl}`: Vector of coefficients.
    - `components_indexes::Dict{String, Vector{Int64}}`: Dictionary containing indexes for different components.

    # Returns
    - `Dict`: Dictionary containing variances for each innovation component.

"""
function get_variances(ε::Vector{Fl}, coefs::Vector{Fl}, components_indexes::Dict{String, Vector{Int64}})::Dict where Fl
    
    variances = Dict()
    for component in ["ξ", "ζ", "ω"]
        !isempty(components_indexes[component]) ? variances[component] = var(coefs[components_indexes[component]]) : nothing
    end
    variances["ε"] = var(ε)
    return variances
end
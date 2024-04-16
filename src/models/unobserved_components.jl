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
ζ_size(T::Int64, stabilize_ζ::Int64)::Int64

    Calculates the size of ζ innovation matrix based on the input T.

    # Arguments
    - `T::Int64`: Length of the original time series.
    - `stabilize_ζ::Int64`: Stabilize parameter ζ.

    # Returns
    - `Int64`: Size of ζ calculated from T.

"""
ζ_size(T::Int64, stabilize_ζ::Int64)::Int64 = T-stabilize_ζ-2

"""
ω_size(T::Int64, s::Int64)::Int64

    Calculates the size of ω innovation matrix based on the input T.

    # Arguments
    - `T::Int64`: Length of the original time series.
    - `s::Int64`: Seasonal period.

    # Returns
    - `Int64`: Size of ω calculated from T.

"""
ω_size(T::Int64, s::Int64, stabilize_ζ::Int64)::Int64 = T - stabilize_ζ - s + 1

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
create_ζ(T::Int64, steps_ahead::Int64, stabilize_ζ::Int64)::Matrix

    Creates a matrix of innovations ζ based on the input sizes, and the desired steps ahead (this is necessary for the forecast function).

    # Arguments
    - `T::Int64`: Length of the original time series.
    - `steps_ahead::Int64`: Number of steps ahead (for estimation purposes this should be set at 0).
    - `stabilize_ζ::Int64`: Stabilize parameter ζ.

    # Returns
    - `Matrix`: Matrix of innovations ζ constructed based on the input sizes.

"""
function create_ζ(T::Int64, steps_ahead::Int64, stabilize_ζ::Int64)::Matrix
    ζ_matrix = Matrix{Float64}(undef, T+steps_ahead, T - 2)
    for t in 1:T+steps_ahead
        ζ_matrix[t, :] = t < T ? vcat(collect(t-2:-1:1), zeros(T-2-length(collect(t-2:-1:1)))) : collect(t-2:-1:t-T+1)
    end
    return ζ_matrix[:, 1:end-stabilize_ζ]
end

"""
create_ω(T::Int64, s::Int64, steps_ahead::Int64)::Matrix

    Creates a matrix of innovations ω based on the input sizes, and the desired steps ahead (this is necessary for the forecast function).

    # Arguments
    - `T::Int64`: Length of the original time series.
    - `s::Int64`: Seasonal period.
    - `steps_ahead::Int64`: Number of steps ahead (for estimation purposes this should be set at 0).
    
    # Returns
    - `Matrix`: Matrix of innovations ω constructed based on the input sizes.

"""
function create_ω(T::Int64, s::Int64, steps_ahead::Int64, stabilize_ζ::Int64)::Matrix
    ω_matrix_size = ω_size(T, s, stabilize_ζ) + stabilize_ζ
    ω_matrix = zeros(T+steps_ahead, ω_matrix_size)
    for t in s+1:T+steps_ahead
        ωₜ_coefs = zeros(ω_matrix_size)
        Mₜ = Int64(floor((t-1)/s))
        lag₁ = [t - j*s for j in 0:Mₜ-1]
        lag₂ = [t - j*s - 1 for j in 0:Mₜ-1]
        ωₜ_coefs[lag₁[lag₁.<=ω_matrix_size+(s - 1)] .- (s - 1)] .= 1
        ωₜ_coefs[lag₂[0 .< lag₂ .<=ω_matrix_size+(s - 1)] .- (s - 1)] .= -1
        ω_matrix[t, :] = ωₜ_coefs
    end
    return ω_matrix[:, 1:end-stabilize_ζ]
end

"""
    create_initial_states_Matrix(T::Int64, s::Int64, steps_ahead::Int64, model_type::String)::Matrix

    Creates an initial states matrix based on the input parameters.

    # Arguments
    - `T::Int64`: Length of the original time series.
    - `s::Int64`: Seasonal period.
    - `steps_ahead::Int64`: Number of steps ahead.
    - `model_type::String`: Type of model.

    # Returns
    - `Matrix`: Initial states matrix constructed based on the input parameters.

"""
function create_initial_states_Matrix(T::Int64, s::Int64, steps_ahead::Int64, model_type::String)::Matrix
    μ₀_coefs = ones(T+steps_ahead)
    ν₀_coefs = vcat([0], collect(1:T+steps_ahead-1))

    if model_type == "Basic Structural"
        γ₀_coefs = zeros(T+steps_ahead, s)
        for t in 1:T+steps_ahead
            γ₀_coefs[t, t % s == 0 ? s : t % s] = 1.0
        end
        return hcat(μ₀_coefs, ν₀_coefs, γ₀_coefs)
    elseif model_type == "Local Linear Trend"
        return hcat(μ₀_coefs, ν₀_coefs)
    elseif model_type == "Local Level"
        return hcat(μ₀_coefs)
    end

end

"""
    create_X_unobserved_components(model_type::String, T::Int64, s::Int64, Exogenous_X::Matrix{Fl}, outlier::Bool, stabilize_ζ::Int64,
             steps_ahead::Int64=0, Exogenous_Forecast::Matrix{Fl}=zeros(steps_ahead, size(Exogenous_X, 2))) where Fl

    Creates the StateSpaceLearning matrix X based on the model type and input parameters.

    # Arguments
    - `model_type::String`: Type of model.
    - `T::Int64`: Length of the original time series.
    - `s::Int64`: Seasonal period.
    - `Exogenous_X::Matrix{Fl}`: Exogenous variables matrix.
    - `outlier::Bool`: Flag for considering outlier component.
    - `stabilize_ζ::Int64`: Stabilize parameter for ζ matrix.
    - `steps_ahead::Int64`: Number of steps ahead (default: 0).
    - `Exogenous_Forecast::Matrix{Fl}`: Exogenous variables forecast matrix (default: zeros).

    # Returns
    - `Matrix`: StateSpaceLearning matrix X constructed based on the input parameters.
"""
function create_X_unobserved_components(model_type::String, T::Int64, s::Int64, Exogenous_X::Matrix{Fl}, outlier::Bool, stabilize_ζ::Int64,
                  steps_ahead::Int64=0, Exogenous_Forecast::Matrix{Fl}=zeros(steps_ahead, size(Exogenous_X, 2))) where Fl

    model_type in ["Basic Structural", "Local Level", "Local Linear Trend"] || error("Model not supported.")

    ξ_matrix = create_ξ(T, steps_ahead)
    ζ_matrix = create_ζ(T, steps_ahead, stabilize_ζ)
    ω_matrix = create_ω(T, s, steps_ahead, stabilize_ζ)
    o_matrix = outlier ? create_o_matrix(T, steps_ahead) : zeros(T+steps_ahead, 0)

    initial_states_matrix = create_initial_states_Matrix(T, s, steps_ahead, model_type)
    if model_type == "Basic Structural"
        return hcat(initial_states_matrix, ξ_matrix, ζ_matrix, ω_matrix, o_matrix, vcat(Exogenous_X, Exogenous_Forecast))
    elseif model_type == "Local Level"
        return hcat(initial_states_matrix, ξ_matrix, o_matrix, vcat(Exogenous_X, Exogenous_Forecast))
    elseif model_type == "Local Linear Trend"
        return hcat(initial_states_matrix, ξ_matrix, ζ_matrix, o_matrix, vcat(Exogenous_X, Exogenous_Forecast))
    end
    
end

"""
    get_components_indexes_unobserved_components(T::Int64, s::Int64, Exogenous_X::Matrix{Fl}, outlier::Bool, model_type::String, stabilize_ζ::Int64)::Dict where Fl

    Generates indexes dict for different components based on the model type and input parameters.

    # Arguments
    - `T::Int64`: Length of the original time series.
    - `s::Int64`: Seasonal period.
    - `Exogenous_X::Matrix{Fl}`: Exogenous variables matrix.
    - `outlier::Bool`: Flag for considering outlier component.
    - `model_type::String`: Type of model.
    - `stabilize_ζ::Int64`: Stabilize parameter for ζ matrix.

    # Returns
    - `Dict`: Dictionary containing the corresponding indexes for each component of the model.

"""
function get_components_indexes_unobserved_components(T::Int64, s::Int64, Exogenous_X::Matrix{Fl}, outlier::Bool, model_type::String, stabilize_ζ::Int64)::Dict where Fl
    
    model_type in ["Basic Structural", "Local Level", "Local Linear Trend"] || error("Model not supported.")

    μ₁_indexes = [1]
    ν₁_indexes = model_type in ["Local Linear Trend", "Basic Structural"] ? [2] : Int64[]
    γ₁_indexes = model_type == "Basic Structural" ? collect(3:2+s) : Int64[]

    initial_states_indexes = μ₁_indexes
    !isempty(ν₁_indexes) ?  initial_states_indexes = vcat(initial_states_indexes, ν₁_indexes) : nothing
    !isempty(γ₁_indexes) ?  initial_states_indexes = vcat(initial_states_indexes, γ₁_indexes) : nothing 

    FINAL_INDEX = initial_states_indexes[end]
    
    ξ_indexes   = collect(FINAL_INDEX + 1:FINAL_INDEX + ξ_size(T))
    FINAL_INDEX = ξ_indexes[end]

    ζ_indexes   = !isempty(ν₁_indexes) ? collect(FINAL_INDEX + 1:FINAL_INDEX + ζ_size(T, stabilize_ζ)) : Int64[]
    FINAL_INDEX = !isempty(ν₁_indexes) ? ζ_indexes[end] : FINAL_INDEX

    ω_indexes   = !isempty(γ₁_indexes) ? collect(FINAL_INDEX + 1:FINAL_INDEX + ω_size(T, s, stabilize_ζ)) : Int64[]
    FINAL_INDEX = !isempty(γ₁_indexes) ? ω_indexes[end] : FINAL_INDEX

    o_indexes   = outlier ? collect(FINAL_INDEX + 1:FINAL_INDEX + o_size(T)) : Int64[]
    FINAL_INDEX = outlier ? o_indexes[end] : FINAL_INDEX

    exogenous_indexes = collect(FINAL_INDEX + 1:FINAL_INDEX + size(Exogenous_X, 2))

    return Dict("μ₁" => μ₁_indexes, "ν₁" => ν₁_indexes, "γ₁" => γ₁_indexes, 
                "ξ" => ξ_indexes, "ζ" => ζ_indexes, "ω" => ω_indexes, "o" => o_indexes, 
                "Exogenous_X" => exogenous_indexes, "initial_states" => initial_states_indexes)
end

"""
    get_variances_unobserved_components(ϵ::Vector{Fl}, coefs::Vector{Fl}, components_indexes::Dict{String, Vector{Int64}})::Dict where Fl

    Calculates variances for each innovation component and for the residuals.

    # Arguments
    - `ϵ::Vector{Fl}`: Vector of residuals.
    - `coefs::Vector{Fl}`: Vector of coefficients.
    - `components_indexes::Dict{String, Vector{Int64}}`: Dictionary containing indexes for different components.

    # Returns
    - `Dict`: Dictionary containing variances for each innovation component.

"""
function get_variances_unobserved_components(ϵ::Vector{Fl}, coefs::Vector{Fl}, components_indexes::Dict{String, Vector{Int64}})::Dict where Fl
    
    variances = Dict()
    for component in ["ξ", "ζ", "ω"]
        !isempty(components_indexes[component]) ? variances[component] = var(coefs[components_indexes[component]]) : nothing
    end
    variances["ϵ"] = var(ϵ)
    return variances
end

"""
    forecast_unobserved_components(output::Output, steps_ahead::Int64, Exogenous_Forecast::Matrix{Fl})::Vector{Float64} where Fl

    Returns the forecast for a given number of steps ahead using the provided StateSpaceLearning output and exogenous forecast data.

    # Arguments
    - `output::Output`: Output object obtained from model fitting.
    - `steps_ahead::Int64`: Number of steps ahead for forecasting.
    - `Exogenous_Forecast::Matrix{Fl}`: Exogenous forecast matrix.
    - `model_dict::Dict`: Dictionary containing the model functions (default: unobserved_components_dict).
    - `exog_model_args::Dict`: Dictionary containing the exogenous model arguments (default: Dict()).

    # Returns
    - `Vector{Float64}`: Vector containing forecasted values.

"""
function forecast_unobserved_components(output::Output, steps_ahead::Int64, Exogenous_Forecast::Matrix{Fl})::Vector{Float64} where Fl
    Exogenous_X = output.X[:, output.components["Exogenous_X"]["Indexes"]]
    complete_matrix = create_X_unobserved_components(output.model_type, output.T, output.s, Exogenous_X, 
                                                     output.outlier, output.stabilize_ζ, steps_ahead, Exogenous_Forecast)

    return complete_matrix[end-steps_ahead+1:end, :]*output.coefs
end

"""
    Dict containing the functions for the unobserved components models.
"""
const unobserved_components_dict = Dict(
    "create_X" => create_X_unobserved_components,
    "create_X_ARGS" => ["model_type", "T", "s", "Exogenous_X", "outlier", "stabilize_ζ"],
    "get_components_indexes" => get_components_indexes_unobserved_components,
    "get_components_indexes_ARGS" => ["T", "s", "Exogenous_X", "outlier", "model_type", "stabilize_ζ"],
    "get_variances" => get_variances_unobserved_components,
    "forecast" => forecast_unobserved_components,
    "forecast_ARGS" => ["output", "steps_ahead", "Exogenous_Forecast"]
)
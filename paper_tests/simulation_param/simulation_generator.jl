using Random, Distributions, Statistics

function generate_series(
    T::Int,
    seed::Union{Nothing,Int}=nothing;
    sparse::Bool=false,
    μ1_mean::Float64=0.0,
    μ1_std::Float64=1.0,
    ν1_mean::Float64=0.0,
    ν1_std::Float64=0.1,
    γ1_mean::Float64=0.0,
    γ1_std::Float64=1.0,
    xi_std::Float64=1.0,
    zeta_std::Float64=0.01,
    omega_std::Float64=1.0,
    eps_std::Float64=0.02,
)
    if seed !== nothing
        Random.seed!(seed)
    end
    s = 12

    # --- 1. Initial State Generation ---
    μ1 = rand(Normal(μ1_mean, μ1_std))
    ν1 = rand(Normal(ν1_mean, ν1_std))
    γ1_initial = rand(Normal(γ1_mean, γ1_std), s - 1)
    γ1_initial .-= mean(γ1_initial)

    # Shocks come from a heavy-tailed Student's t-distribution
    level_shock_dist = Normal(0, xi_std)
    trend_shock_dist = Normal(0, zeta_std)
    seasonal_shock_dist = Normal(0, omega_std)

    # Initialize and store the TRUE innovations and components
    true_innovations = Dict(
        "level" => zeros(T), "trend" => zeros(T), "seasonal" => zeros(T)
    )
    μ = [μ1];
    ν = [ν1];
    γ_vec = vcat(γ1_initial, -sum(γ1_initial))

    # --- 2. Generate Series with Dense, Heavy-Tailed Shocks ---
    for t in 2:T
        # Shocks are now generated at every time step
        if sparse
            level_shock = rand(level_shock_dist) * rand(Bernoulli(0.03))
            trend_shock = rand(trend_shock_dist) * rand(Bernoulli(0.03))
            seasonal_shock = rand(seasonal_shock_dist) * rand(Bernoulli(0.03))
        else
            level_shock = rand(level_shock_dist)
            trend_shock = rand(trend_shock_dist)
            seasonal_shock = rand(seasonal_shock_dist)
        end

        # Store the true innovations
        true_innovations["level"][t] = level_shock
        true_innovations["trend"][t] = trend_shock
        true_innovations["seasonal"][t] = seasonal_shock

        # Update components
        push!(ν, ν[t - 1] + trend_shock)
        push!(μ, μ[t - 1] + ν[t - 1] + level_shock)
        if t > s
            push!(γ_vec, -sum(γ_vec[(t - s + 1):(t - 1)]) + seasonal_shock)
        end
    end

    # Final "true" series (before observation noise)
    # μ = μ .- μ[1]
    # ν = ν .- ν[1]
    # γ_vec = γ_vec .- mean(γ_vec[1:s])

    y_observed = [μ[t] + γ_vec[t] + rand(Normal(0, eps_std)) for t in 1:T]

    return y_observed, μ, ν, γ_vec, xi_std, zeta_std, omega_std, eps_std
end

using Random, Distributions, Statistics
using Polynomials

function ar_polinomial(p::Vector{Fl}) where {Fl}
    return Polynomial([one(Fl); -p])
end

function ma_polinomial(q::Vector{Fl}) where {Fl}
    return Polynomial([one(Fl); q])
end

function roots_of_inverse_polinomial(poly::Polynomial)
    return roots(poly) .^ -1
end

function assert_stationarity(p::Vector{Fl}) where {Fl}
    poly = ar_polinomial(p)
    return all(abs.(roots_of_inverse_polinomial(poly)) .< 1)
end

function assert_invertibility(q::Vector{Fl}) where {Fl}
    poly = ma_polinomial(q)
    return all(abs.(roots_of_inverse_polinomial(poly)) .< 1)
end

function generate_sarima_exog(T::Int, M::Int)
    X = zeros(T, M)
    s = 12
    for j in 1:M
        try_again = true
        ar_params = nothing
        ma_params = nothing
        sar_params = nothing
        sma_params = nothing
        p = nothing
        q = nothing
        P = nothing
        Q = nothing

        while try_again
            p = rand([0, 1, 2, 3])
            q = rand([0, 1, 2, 3])
            P = rand([0, 1])
            Q = rand([0, 1])

            ar_params = rand(Normal(0, 0.2), p)
            ma_params = rand(Normal(0, 0.2), q)
            sar_params = rand(Normal(0, 0.2), P)
            sma_params = rand(Normal(0, 0.2), Q)

            ar_t = vcat(ar_params, zeros(s - p - 1), sar_params)
            ma_t = vcat(ma_params, zeros(s - q - 1), sma_params)

            if assert_stationarity(ar_t) && assert_invertibility(ma_t)
                try_again = false
            end
        end

        data = vcat(
            rand(Normal(0, 1), max(p, q, P * s, Q * s)), zeros(T - max(p, q, P * s, Q * s))
        )
        for i in (max(p, q, P * s, Q * s) + 1):T
            seasonal_comp = 0
            if i > s
                P_term = if P == 0
                    0
                else
                    sum(sar_params .* data[(i - s * P):(i - s):(i - s * P + 1)])
                end
                Q_term = if Q == 0
                    0
                else
                    sum(sma_params .* data[(i - s * Q):(i - s):(i - s * Q + 1)])
                end
                seasonal_comp = P_term + Q_term
            end
            p_term = sum(ar_params .* data[(i - p):(i - 1)])
            q_term = sum(ma_params .* data[(i - q):(i - 1)])
            data[i] += p_term + q_term + seasonal_comp
        end

        # Add Gaussian noise
        data += rand(Normal(0, 3), T)

        X[:, j] = data
    end
    return X
end

function generate_subset(T::Int, M::Int, K::Int)
    s = 12

    μ1 = 1.0
    ν1 = 0.001
    γ1 = [1.5, 2.6, 3.0, 2.6, 1.5, 0.0, -1.5, -2.6, -3.0, -2.6, -1.5]

    stds = abs.([rand(Normal(0, 0.5)), rand(Normal(0, 0.001)), rand(Normal(0, 0.3))])
    μ = [μ1]
    ν = [ν1]
    γ = [γ1...]
    y = [μ1 + γ1[1]]

    for t in 2:T
        push!(ν, ν[t - 1] + rand(Normal(0, stds[2])))
        push!(μ, μ[t - 1] + ν[t - 1] + rand(Normal(0, stds[1])))
        if t > 11
            push!(γ, -sum(γ[t - j] for j in 1:(s - 1)) + stds[3])
        end
        push!(y, μ[t] + γ[t])
    end

    X = generate_sarima_exog(T, M)
    true_exps = collect(1:K)
    β = ones(length(true_exps))#rand(Normal(0, 1.0), length(true_exps))
    X_exp = X[:, true_exps] * β
    y = y + X_exp
    y += rand(Normal(0, 0.2), T)
    return y, true_exps, X, vcat(β, zeros(M - K))
end

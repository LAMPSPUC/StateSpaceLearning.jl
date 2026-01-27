
function align_components!(
    μ_hat::AbstractVector{<:AbstractFloat},
    ν_hat::AbstractVector{<:AbstractFloat},
    γ_hat::AbstractVector{<:AbstractFloat},
    μ_true::AbstractVector{<:AbstractFloat},
    ν_true::AbstractVector{<:AbstractFloat},
    γ_true::AbstractVector{<:AbstractFloat},
)
    μ_offset = μ_true[1] - μ_hat[1]
    ν_offset = ν_true[1] - ν_hat[1]
    γ_offset = mean(γ_true) - mean(γ_hat)

    μ_hat .+= μ_offset
    ν_hat .+= ν_offset
    γ_hat .+= γ_offset

    return μ_hat, ν_hat, γ_hat
end

function get_SSL_results(
    y_train::Vector{Fl}, s::Int, μ_true, ν_true, γ_true, inf_criteria::String
) where {Fl<:AbstractFloat}
    min_y = minimum(y_train)
    max_y = maximum(y_train)
    normy = (y_train .- min_y) ./ (max_y - min_y)

    model = StateSpaceLearning.StructuralModel(
        normy; freq_seasonal=s, outlier=false, ξ_threshold=0, ζ_threshold=1, ω_threshold=1
    )
    StateSpaceLearning.fit!(
        model;
        information_criteria=inf_criteria,
        ϵ=0.05,
        penalize_initial_states=true,
        α=0.1,
    )

    μ_hat = model.output.decomposition["trend"] .* (max_y - min_y)
    ν_hat = model.output.decomposition["slope"] .* (max_y - min_y)
    γ_hat = model.output.decomposition["seasonal_$s"] .* (max_y - min_y)

    align_components!(μ_hat, ν_hat, γ_hat, μ_true, ν_true, γ_true)

    return μ_hat, ν_hat, γ_hat
end

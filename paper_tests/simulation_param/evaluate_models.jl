function get_SSL_results(
    y_train::Vector{Fl}, s::Int, μ_true, ν_true, γ_true, inf_criteria::String
) where {Fl<:AbstractFloat}

    model = StateSpaceLearning.StructuralModel(
        y_train; freq_seasonal=s, outlier=false, ξ_threshold=0, ζ_threshold=1, ω_threshold=1
    )
    StateSpaceLearning.fit!(
        model;
        information_criteria=inf_criteria,
        ϵ=0.05,
        penalize_initial_states=true,
        α=0.1,
    )

    μ_hat = model.output.decomposition["trend"]
    ν_hat = model.output.decomposition["slope"]
    γ_hat = model.output.decomposition["seasonal_$s"]

    return μ_hat, ν_hat, γ_hat
end

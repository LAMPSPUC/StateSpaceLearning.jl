function get_confusion_matrix(selected, true_features, false_features)
    all_features = vcat(true_features, false_features)
    true_positives = count(i -> i in true_features, selected)
    false_positives = count(i -> i in false_features, selected)
    false_negatives = count(i -> i in true_features, all_features) - true_positives
    true_negatives = count(i -> i in false_features, all_features) - false_positives
    return true_positives, false_positives, false_negatives, true_negatives
end

function get_SSL_results(
    y_train::Vector{Fl},
    true_features::Vector{Int},
    false_features::Vector{Int},
    X_train::Matrix{Fl},
    inf_criteria::String,
    true_β::Vector{Fl},
) where {Fl<:AbstractFloat}
    series_result = nothing

    model = StateSpaceLearning.StructuralModel(
        y_train;
        level=true,
        stochastic_level=true,
        trend=true,
        stochastic_trend=true,
        seasonal=true,
        stochastic_seasonal=true,
        freq_seasonal=12,
        outlier=false,
        ζ_ω_threshold=12,
        exog=X_train,
    )
    t = @elapsed StateSpaceLearning.fit!(
        model;
        α=1.0,
        information_criteria=inf_criteria,
        ϵ=0.05,
        penalize_exogenous=true,
        penalize_initial_states=true,
    )

    selected = model.output.components["exog"]["Selected"]
    true_positives, false_positives, false_negatives, true_negatives = get_confusion_matrix(
        selected, true_features, false_features
    )

    mse = mse_func(model.output.components["exog"]["Coefs"], true_β)
    bias = bias_func(model.output.components["exog"]["Coefs"], true_β)

    series_result = DataFrame(
        [
            [t],
            [mse],
            [bias],
            [true_positives],
            [false_positives],
            [false_negatives],
            [true_negatives],
        ],
        [
            :time,
            :mse,
            :bias,
            :true_positives,
            :false_positives,
            :false_negatives,
            :true_negatives,
        ],
    )

    return series_result
end

function get_SS_res_results(
    y_train::Vector{Fl},
    true_features::Vector{Int},
    false_features::Vector{Int},
    X_train::Matrix{Fl},
    inf_criteria::String,
    true_β::Vector{Fl},
) where {Fl<:AbstractFloat}
    py"""
    import math
    import statsmodels.api as sm
    import numpy as np
    def evaluate_ss(y_train):
        model_components = {'irregular': True, 'level': True, 'trend': True, 'freq_seasonal': [{'period': 12}], 'stochastic_level': True, 'stochastic_trend': True, 'stochastic_freq_seasonal': [True]}
        model = sm.tsa.UnobservedComponents(np.array(y_train), **model_components)
        results = model.fit(disp = False, maxiter = 1e5)
        trend_component = results.level["smoothed"] + results.trend["smoothed"]
        seasonal_component = results.freq_seasonal[0]["smoothed"]
        deseasonalized_detrended_series = np.array(y_train) - trend_component - seasonal_component
        converged = results.mle_retvals['converged']
        return deseasonalized_detrended_series, converged
    """
    t = @elapsed begin
        res, converged = py"evaluate_ss"(y_train)

        lasso_path = glmnet(X_train, res; alpha=1.0, intercept=false)
        coefs1, _ = StateSpaceLearning.get_path_information_criteria(
            lasso_path, X_train, res, inf_criteria; intercept=false
        )
        penalty_factor = 1 ./ (abs.(coefs1))
        lasso_path2 = glmnet(
            X_train, res; alpha=1.0, penalty_factor=penalty_factor, intercept=false
        )
        lasso_coefs, _ = StateSpaceLearning.get_path_information_criteria(
            lasso_path2, X_train, res, inf_criteria; intercept=false
        )
    end
    selected = findall(i -> i != 0, lasso_coefs)

    true_positives, false_positives, false_negatives, true_negatives = get_confusion_matrix(
        selected, true_features, false_features
    )

    mse = mse_func(lasso_coefs, true_β)
    bias = bias_func(lasso_coefs, true_β)

    series_result = DataFrame(
        [
            [t],
            [mse],
            [bias],
            [true_positives],
            [false_positives],
            [false_negatives],
            [true_negatives],
        ],
        [
            :time,
            :mse,
            :bias,
            :true_positives,
            :false_positives,
            :false_negatives,
            :true_negatives,
        ],
    )

    return series_result, converged
end

function get_exogenous_ss_inf_criteria(
    y_train::Vector{Fl}, X_train::Matrix{Fl}
) where {Fl<:AbstractFloat}
    py"""
    import math
    import statsmodels.api as sm
    import numpy as np
    def evaluate_ss(y_train, X_train):
        model_components = {'irregular': True, 'exog': X_train, 'level': True, 'trend': True, 'freq_seasonal': [{'period': 12}], 'stochastic_level': True, 'stochastic_trend': True, 'stochastic_freq_seasonal': [True]}
        model = sm.tsa.UnobservedComponents(np.array(y_train), **model_components)
        results = model.fit(disp = False, maxiter = 1e5)
        aic = results.aic
        bic = results.bic
        coefs_exog = results.params
        converged = results.mle_retvals['converged']
        return aic, bic, coefs_exog, converged
    """
    return py"evaluate_ss"(y_train, X_train)
end

function get_forward_ss(
    y_train::Vector{Fl},
    true_features::Vector{Int},
    false_features::Vector{Int},
    X_train::Matrix{Fl},
    inf_criteria::String,
    true_β::Vector{Fl},
) where {Fl<:AbstractFloat}
    best_inf_crit = Inf
    current_inf_crit = 0
    coefs = nothing
    selected = []
    remaining_exogs = collect(1:size(X_train, 2))
    stop = false
    last_converged = nothing
    converged = nothing
    t = @elapsed begin
        while !stop
            iteration_inf_vec = []
            iteration_coefs_vec = []
            for i in remaining_exogs
                aic, bic, coefs_exog, converged = get_exogenous_ss_inf_criteria(
                    y_train, X_train[:, sort(vcat(selected, i))]
                )
                inf_crit = inf_criteria == "aic" ? aic : bic
                push!(iteration_inf_vec, inf_crit)
                push!(
                    iteration_coefs_vec,
                    coefs_exog[(end - length(vcat(selected, i)) + 1):end],
                )
            end
            current_inf_crit = minimum(iteration_inf_vec)
            if current_inf_crit < best_inf_crit
                best_inf_crit = current_inf_crit
                best_exog = argmin(iteration_inf_vec)
                coefs = iteration_coefs_vec[best_exog]
                push!(selected, best_exog)
                remaining_exogs = setdiff(remaining_exogs, best_exog)
                last_converged = converged
            else
                stop = true
            end
        end
    end
    estimated_coefs = zeros(size(X_train, 2))
    estimated_coefs[sort(selected)] = coefs

    true_positives, false_positives, false_negatives, true_negatives = get_confusion_matrix(
        selected, true_features, false_features
    )

    mse = mse_func(estimated_coefs, true_β)
    bias = bias_func(estimated_coefs, true_β)

    series_result = DataFrame(
        [
            [t],
            [mse],
            [bias],
            [true_positives],
            [false_positives],
            [false_negatives],
            [true_negatives],
        ],
        [
            :time,
            :mse,
            :bias,
            :true_positives,
            :false_positives,
            :false_negatives,
            :true_negatives,
        ],
    )

    return series_result, last_converged
end

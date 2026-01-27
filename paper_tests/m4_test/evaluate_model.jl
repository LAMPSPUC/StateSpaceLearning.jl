function evaluate_SSL(
    initialization_df::DataFrame,
    results_df::DataFrame,
    input::Dict,
    outlier::Bool,
    α::AbstractFloat,
    selection::String,
    information_criteria::String,
    param::Dict,
)
    normalized_y = input["normalized_train"]
    y_train = input["train"]
    y_test = input["test"]
    max_y = input["max"]
    min_y = input["min"]

    T = length(normalized_y)
    if param["sample_size"] != "all"
        sample_size = param["sample_size"]
        normalized_y = normalized_y[max(1, T - sample_size + 1):end]
    end

    if outlier
        ξ_threshold = param["ξ_threshold"]
    else
        ξ_threshold = 0
    end
    
    model = StateSpaceLearning.StructuralModel(
        normalized_y;
        level="stochastic",
        slope="stochastic",
        seasonal=param["seasonal"],
        freq_seasonal=param["freq_seasonal"],
        cycle=param["cycle"],
        cycle_period=param["cycle_period"],
        outlier=outlier,
        ξ_threshold=ξ_threshold,
        ζ_threshold=param["ζ_threshold"],
        ω_threshold=param["ω_threshold"],
        ϕ_threshold=param["ϕ_threshold"]
    )

    if selection == "split"
        StateSpaceLearning.fit_split!(
            model;
            H=param["H"],
            α_set=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
            information_criteria=information_criteria,
            ϵ=0.05,
            penalize_exogenous=true,
            penalize_initial_states=true,
        )
    else
        StateSpaceLearning.fit!(
            model;
            α=α,
            information_criteria=information_criteria,
            ϵ=0.05,
            penalize_exogenous=true,
            penalize_initial_states=true,
        )
    end

    normalized_prediction = StateSpaceLearning.forecast(model, param["H"])
    prediction = de_normalize(normalized_prediction, max_y, min_y)

    normalized_scenarios = StateSpaceLearning.simulate(model, param["H"], 1000)
    scenarios = de_normalize(normalized_scenarios, max_y, min_y)

    mase = MASE(y_train, y_test, prediction; m=param["m"])
    smape = sMAPE(y_test, prediction)
    crps = CRPS(scenarios, y_test)

    results_df = vcat(
        results_df, DataFrame([[mase], [smape], [crps]], [:MASE, :sMAPE, :CRPS])
    )
    return initialization_df, results_df
end


function evaluate_SS(input::Dict, m::Int, H::Int, frequency::Int)
    """
    Evaluate statsmodels UnobservedComponents model using PyCall
    Similar to evaluate_ss in m4_test.py
    Requires PyCall to be imported in the calling scope
    """
    
    y_train = input["train"]
    y_test = input["test"]

    y_train_py = PyObject(y_train)
    y_test_py = PyObject(y_test)
    H_py = PyObject(H)
    frequency_py = PyObject(frequency)
    
    py"""
    import statsmodels.api as sm
    import numpy as np
    
    def evaluate_ss_py(y_train, y_test, H, frequency):
        # Prepare model components
        if frequency > 1:
            model_components = {
                'irregular': True, 
                'level': True, 
                'trend': True, 
                'seasonal': frequency,
                'stochastic_level': True, 
                'stochastic_trend': True, 
                'stochastic_seasonal': True
            }
        else:
            model_components = {
                'irregular': True, 
                'level': True, 
                'trend': True,
                'stochastic_level': True, 
                'stochastic_trend': True
            }
        
        # Create and fit model
        model = sm.tsa.UnobservedComponents(np.array(y_train), **model_components)

        results = model.fit(disp=False, maxiter=1e5)
        
        # Get forecast
        config = {
            'repetitions': 1000,
            'steps': H
        }
        forecast_obj = results.get_forecast(**config)
        forecast_df = forecast_obj.summary_frame()
        
        # Generate simulations (300 per time step as in Python code)
        simulation = np.empty((len(forecast_df), 1000))
        for i in range(len(forecast_df)):
            simulation[i] = np.random.normal(
                forecast_df["mean"].values[i], 
                forecast_df["mean_se"].values[i], 
                size=1000
            )
        
        # Denormalize
        forecast_values = forecast_df["mean"].values
        
        return forecast_values, simulation
    """
    
    prediction, scenarios = py"evaluate_ss_py"(
        y_train_py, y_test_py, 
        H_py, frequency_py
    )
    
    # Convert back to Julia arrays
    prediction = Vector{Float64}(prediction)
    scenarios = Matrix{Float64}(scenarios)
    
    # Calculate metrics
    mase = MASE(y_train, y_test, prediction; m = m)
    smape = sMAPE(y_test, prediction)
    crps = CRPS(scenarios, y_test)
    
    return DataFrame([[mase], [smape], [crps]], [:MASE, :sMAPE, :CRPS])
    
end
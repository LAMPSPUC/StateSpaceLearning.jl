# StateSpaceLearning

| **Build Status** | **Coverage** | **Documentation** | **CodeStyle** |
|:-----------------:|:-----------------:|:-----------------:|:-----------------:|
| [![ci](https://github.com/LAMPSPUC/StateSpaceLearning.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/LAMPSPUC/StateSpaceLearning.jl/actions/workflows/ci.yml) | [![codecov](https://codecov.io/gh/LAMPSPUC/StateSpaceLearning.jl/graph/badge.svg?token=VDpuXvPSI2)](https://codecov.io/gh/LAMPSPUC/StateSpaceLearning.jl) | [![](https://img.shields.io/badge/docs-latest-blue.svg)]( https://lampspuc.github.io/StateSpaceLearning.jl/) | [![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

StateSpaceLearning.jl is a package for modeling and forecasting time series in a high-dimension regression framework.

## Quickstart

```julia
using StateSpaceLearning

y = randn(100)

# Instantiate Model
model = StructuralModel(y)

# Fit Model
fit!(model)

# Point Forecast
prediction = forecast(model, 12) # Gets a 12 steps ahead prediction

# Scenarios Path Simulation
simulation = simulate(model, 12, 1000) # Gets 1000 scenarios path of 12 steps ahead predictions
```

## StructuralModel Arguments

* `y::Union{Vector,Matrix}`: Time series data
* `level::String`: Level component type: "stochastic", "deterministic", or "none" (default: "stochastic")
* `slope::String`: Slope component type: "stochastic", "deterministic", or "none" (default: "stochastic")
* `seasonal::String`: Seasonal component type: "stochastic", "deterministic", or "none" (default: "stochastic")
* `cycle::String`: Cycle component type: "stochastic", "deterministic", or "none" (default: "none")
* `freq_seasonal::Union{Int,Vector{Int}}`: Seasonal frequency or vector of frequencies (default: 12)
* `cycle_period::Union{Union{Int,<:AbstractFloat},Vector{Int},Vector{<:AbstractFloat}}`: Cycle period or vector of periods (default: 0)
* `outlier::Bool`: Include outlier component (default: true)
* `ξ_threshold::Int`: Threshold for level innovations (default: 1)
* `ζ_threshold::Int`: Threshold for slope innovations (default: 12)
* `ω_threshold::Int`: Threshold for seasonal innovations (default: 12)
* `ϕ_threshold::Int`: Threshold for cycle innovations (default: 12)
* `stochastic_start::Int`: Starting point for stochastic components (default: 1)
* `exog::Matrix`: Matrix of exogenous variables (default: zeros(length(y), 0))
* `dynamic_exog_coefs::Union{Vector{<:Tuple}, Nothing}`: Dynamic exogenous coefficients (default: nothing)

## Features

Current features include:
* Model estimation using elastic net based regularization
* Automatic component decomposition (trend, seasonal, cycle)
* Point forecasting and scenario simulation
* Missing value imputation
* Outlier detection and robust modeling
* Multiple seasonal frequencies support
* Deterministic and stochastic component options
* Dynamic exogenous variable handling
* Best subset selection for exogenous variables

## Quick Examples

### Fitting, forecasting and simulating
Quick example of fit and forecast for the air passengers time-series.

```julia
using StateSpaceLearning
using CSV
using DataFrames
using Plots

airp = CSV.File(StateSpaceLearning.AIR_PASSENGERS) |> DataFrame
log_air_passengers = log.(airp.passengers)
steps_ahead = 30

model = StructuralModel(log_air_passengers)
fit!(model)
prediction_log = forecast(model, steps_ahead) # arguments are the output of the fitted model and number of steps ahead the user wants to forecast
prediction = exp.(prediction_log)

plot_point_forecast(airp.passengers, prediction)
```
![quick_example_airp](./docs/src/assets/quick_example_airp.PNG)

```julia
N_scenarios = 1000
simulation = simulate(model, steps_ahead, N_scenarios) # arguments are the output of the fitted model, number of steps ahead the user wants to forecast and number of scenario paths

plot_scenarios(airp.passengers, exp.(simulation))

```
![airp_sim](./docs/src/assets/airp_sim.svg)

### Component Extraction
Quick example on how to perform component extraction in time series utilizing StateSpaceLearning.

```julia
using CSV
using DataFrames
using Plots

airp = CSV.File(StateSpaceLearning.AIR_PASSENGERS) |> DataFrame
log_air_passengers = log.(airp.passengers)

model = StructuralModel(log_air_passengers)
fit!(model)

# Access decomposed components directly
trend = model.output.decomposition["trend"]
seasonal = model.output.decomposition["seasonal_12"]

plot(trend, w=2, color = "Black", lab = "Trend Component", legend = :outerbottom)
plot(seasonal, w=2, color = "Black", lab = "Seasonal Component", legend = :outerbottom)
```

| ![quick_example_trend](./docs/src/assets/trend.svg) | ![quick_example_seas](./docs/src/assets/seasonal.svg)|
|:------------------------------:|:-----------------------------:|


### Best Subset Selection and Dynamic Coefficients
Example of performing best subset selection and using dynamic coefficients:

```julia
using StateSpaceLearning
using CSV
using DataFrames
using Random

Random.seed!(2024)

# Load data
airp = CSV.File(StateSpaceLearning.AIR_PASSENGERS) |> DataFrame
log_air_passengers = log.(airp.passengers)

# Create exogenous features
X = rand(length(log_air_passengers), 10)
β = rand(3)
y = log_air_passengers + X[:, 1:3]*β

# Create model with exogenous variables
model = StructuralModel(y; 
    exog = X
)

# Fit model with elastic net regularization
fit!(model; 
    α = 1.0, # 1.0 for Lasso, 0.0 for Ridge
    information_criteria = "bic",
    ϵ = 0.05,
    penalize_exogenous = true,
    penalize_initial_states = true
)

# Get selected features
selected_exog = model.output.components["exog"]["Selected"]
```

### Missing values imputation
Quick example of completion of missing values for the air passengers time-series (artificial NaN values are added to the original time-series).

```julia
using CSV
using DataFrames
using Plots

airp = CSV.File(StateSpaceLearning.AIR_PASSENGERS) |> DataFrame
log_air_passengers = log.(airp.passengers)

airpassengers = AbstractFloat.(airp.passengers)
log_air_passengers[60:72] .= NaN

model = StructuralModel(log_air_passengers)
fit!(model)

fitted_completed_missing_values = ones(144).*NaN; fitted_completed_missing_values[60:72] = exp.(model.output.fitted[60:72])
real_removed_valued = ones(144).*NaN; real_removed_valued[60:72] = deepcopy(airp.passengers[60:72])
airpassengers[60:72] .= NaN

plot(airpassengers, w=2 , color = "Black", lab = "Historical", legend = :outerbottom)
plot!(real_removed_valued, lab = "Real Removed Values", w=2, color = "red")
plot!(fitted_completed_missing_values, lab = "Fit in Sample completed values", w=2, color = "blue")

```
![quick_example_completion_airp](./docs/src/assets/quick_example_completion_airp.PNG)

### Outlier Detection
Quick example of outlier detection for an altered air passengers time-series (artificial NaN values are added to the original time-series).

```julia
using CSV
using DataFrames
using Plots

airp = CSV.File(StateSpaceLearning.AIR_PASSENGERS) |> DataFrame
log_air_passengers = log.(airp.passengers)

log_air_passengers[60] = 10
log_air_passengers[30] = 1
log_air_passengers[100] = 2

model = StructuralModel(log_air_passengers)
fit!(model)

detected_outliers = findall(i -> i != 0, model.output.components["o"]["Coefs"])

plot(log_air_passengers, w=2 , color = "Black", lab = "Historical", legend = :outerbottom)
scatter!([detected_outliers], log_air_passengers[detected_outliers], lab = "Detected Outliers")

```
![quick_example_completion_airp](./docs/src/assets/outlier.svg)

### StateSpaceModels initialization
Quick example on how to use StateSpaceLearning to initialize  StateSpaceModels

```julia
using CSV
using DataFrames
using StateSpaceModels

airp = CSV.File(StateSpaceLearning.AIR_PASSENGERS) |> DataFrame
log_air_passengers = log.(airp.passengers)

model = StructuralModel(log_air_passengers)
fit!(model)

residuals_variances = model.output.residuals_variances

ss_model = BasicStructural(log_air_passengers, 12)
StateSpaceModels.set_initial_hyperparameters!(ss_model, Dict("sigma2_ε" => residuals_variances["ε"], 
                                         "sigma2_ξ" =>residuals_variances["ξ"], 
                                         "sigma2_ζ" =>residuals_variances["ζ"], 
                                         "sigma2_ω" =>residuals_variances["ω_12"]))
StateSpaceModels.fit!(ss_model)
```

## Paper Results Reproducibility

The paper has two experiments (results for the M4 competition and a simulation study). To reproduce each experiment follow the instructions below:

### M4 Experiment

To reproduce M4 paper results you can clone the repository and run the following commands on terminal:

```shell
julia paper_tests/m4_test/m4_test.jl
python paper_tests/m4_test/m4_test.py
```

The results for SSL model in terms of MASE and sMAPE for all 48000 series will be stored in folder "paper_tests/m4_test/results_SSL". The average results of MASE, sMAPE and OWA will be saved in file "paper_tests/m4_test/metric_results/SSL_METRICS_RESULTS.csv".

The results for SS model in terms of MASE and sMAPE for all 48000 series will be stored in folder "paper_tests/m4_test/results_SS". The average results of MASE, sMAPE and OWA will be saved in file "paper_tests/m4_test/metric_results/SS_METRICS_RESULTS.csv".

### Simulation Experiment

To reproduce the simulation results you can clone the repository and run the following commands on terminal:

```shell
julia paper_tests/simulation_test/simulation.jl 0
```

As this test takes a long time, you may want to run it in parallel, for that you can change the last argument to be number of workers to use in the parallelization:

```shell
julia paper_tests/simulation_test/simulation.jl 3
```

The results will be saved in two separated files: "paper_tests/simulation_test/results_metrics/metrics_confusion_matrix.csv" and "paper_tests/simulation_test/results_metrics/metrics_bias_mse.csv"


## Contributing

* PRs such as adding new models and fixing bugs are very welcome!
* For nontrivial changes, you'll probably want to first discuss the changes via issue.

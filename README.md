# StateSpaceLearning

| **Build Status** | **Coverage** |
|:-----------------:|:-----------------:|
| [![ci](https://github.com/LAMPSPUC/StateSpaceLearning/actions/workflows/ci.yml/badge.svg)](https://github.com/LAMPSPUC/StateSpaceLearning/actions/workflows/ci.yml) | [![codecov](https://codecov.io/gh/LAMPSPUC/StateSpaceLearning/graph/badge.svg?token=VDpuXvPSI2)](https://codecov.io/gh/LAMPSPUC/StateSpaceLearning) |


StateSpaceLearning.jl is a package for modeling and forecasting time series in a high-dimension regression framework.

## Quickstart

```julia
using StateSpaceLearning

y = randn(100)

#Fit Model
output = StateSpaceLearning.fit_model(y)

#Main output options 
model_type = output.model_type # State Space Equivalent model utilized in the estimation (default = Basic Structural).
X                   = output.X # High Dimension Regression utilized in the estimation.
coefs               = output.coefs # High Dimension Regression coefficients estimated in the estimation.
ϵ                   = output.ϵ # Residuals of the model.
fitted              = output.fitted # Fit in Sample of the model.
components          = output.components # Dictionary containing information about each component of the model, each component has the keys: "Values" (The value of the component in each timestamp) , "Coefs" (The coefficients estimated for each element of the component) and "Indexes" (The indexes of the elements of the component in the high dimension regression "X").
residuals_variances = output.residuals_variances # Dictionary containing the estimated variances for the innovations components (that is the information that can be utilized to initialize the state space model).
s                   = ouotput.s # The seasonal frequency utilized in the model (default = 12).
T                   = output.T # The length of the original time series.
outlier             = output.outlier # Boolean indicating the presence of outlier component (default = false).
valid_indexes       = output.valid_indexes # Vector containing valid indexes of the time series (non valid indexes represent NaN values in the time series).
stabilize_ζ         = output.stabilize_ζ # Stabilize_ζ parameter (default = 0). A non 0 value for this parameter might be important in terms of forecast for some time series to lead to more stable predictions (we recommend stabilize_ζ = 11 for monthly series).

#Forecast
prediction = StateSpaceLearning.forecast(output, 12) #Gets a 12 steps ahead prediction

```

## Features

Current features include:
* Estimation
* Components decomposition
* Forecasting
* Completion of missing values
* Predefined models, including:
  * Basic Structural"
  * Local Linear Trend
  * Local Level

## Quick Examples

### Fitting and forecasting
Quick example of fit and forecast for the air passengers time-series.

```julia
using CSV
using DataFrames
using Plots
using StateSpaceModels

airp = CSV.File(StateSpaceModels.AIR_PASSENGERS) |> DataFrame
log_air_passengers = log.(airp.passengers)
steps_ahead = 30

output = StateSpaceLearning.fit_model(log_air_passengers)
prediction_raw = StateSpaceLearning.forecast(output, steps_ahead)
prediction = exp.(prediction_raw)

plot(airp.passengers, w=2 , color = "Black", lab = "Historical", legend = :outerbottom)
plot!(vcat(ones(output.T).*NaN, prediction), lab = "Forcast", w=2, color = "blue")
```
![quick_example_airp](./docs/assets/quick_example_airp.png)

### Completion of missing values
Quick example of completion of missing values for the air passengers time-series (artificial NaN values are added to the original time-series).

```julia
using CSV
using DataFrames
using Plots
using StateSpaceModels

airp = CSV.File(StateSpaceModels.AIR_PASSENGERS) |> DataFrame
airpassengers = (airp.passengers).*1.0
log_air_passengers = log.(airpassengers)
steps_ahead = 30

log_air_passengers[60:72] .= NaN

output = StateSpaceLearning.fit_model(log_air_passengers)

fitted_completed_missing_values = ones(144).*NaN; fitted_completed_missing_values[60:72] = exp.(output.fitted[60:72])
real_removed_valued = ones(144).*NaN; real_removed_valued[60:72] = deepcopy(airpassengers[60:72])
airpassengers[60:72] .= NaN

plot(airpassengers, w=2 , color = "Black", lab = "Historical", legend = :outerbottom)
plot!(real_removed_valued, lab = "Real Removed Values", w=2, color = "red")
plot!(fitted_completed_missing_values, lab = "Fit in Sample completed values", w=2, color = "blue")

```
![quick_example_completion_airp](./docs/assets/quick_example_completion_airp.png)

## Contributing

* PRs such as adding new models and fixing bugs are very welcome!
* For nontrivial changes, you'll probably want to first discuss the changes via issue.
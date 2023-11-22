# Manual

## Quick Start Guide

Although StateSpaceLearning.jl has a lot of functionalities, different models and interfaces 
users usuallly just want to fit a model and analyse the residuals, components and make some forecasts.
The following code is a quick start to perform these tasks

```julia
using StateSpaceLearning
using Plots

y = randn(100)

output = StateSpaceLearning.fit_model(y)

prediction = StateSpaceLearning.forecast(output, 12)

plot(y, w=2 , color = "Black", lab = "Historical", legend = :outerbottom)
plot!(output.fitted, w=2 , color = "Black", lab = "Fit In Sample", legend = :outerbottom)
plot!(vcat(ones(output.T).*NaN, prediction), lab = "Forcast", w=2, color = "blue")

```
### Completion of missing values
Quick example of completion of missing values for the air passengers time-series (artificial NaN values are added to the original time-series).

```julia

y = rand(144)

y[60:72] .= NaN

output = StateSpaceLearning.fit_model(y)

completed_values = output.fitted[60:72]

```

### Component extraction
Quick example of component extraction in a time series

```julia
using CSV
using DataFrames
using Plots
using StateSpaceModels

airp = CSV.File(StateSpaceModels.AIR_PASSENGERS) |> DataFrame
log_air_passengers = log.(airp.passengers)

output = StateSpaceLearning.fit_model(log_air_passengers)

μ₁ = output.components["μ₁"]["Values"]
ν₁ = output.components["ν₁"]["Values"]
γ₁ = output.components["γ₁"]["Values"]
ξ  = output.components["ξ"]["Values"]
ζ  = output.components["ζ"]["Values"]
ω  = output.components["ω"]["Values"]

```

### Models

The package provides a variaty of pre-defined models. If there is any model that you wish was in the package, feel free to open an issue or pull request.

### Basic Structural Model
The basic structural state-space model consists of a trend (level + slope) and a seasonal
component. It is defined by:

```math
\begin{gather*}
    \begin{aligned}
        y_{t} &=  \mu_{t} + \gamma_{t} + \varepsilon_{t} \quad &\varepsilon_{t} \sim \mathcal{N}(0, \sigma^2_{\varepsilon})\\
        \mu_{t+1} &= \mu_{t} + \nu_{t} + \xi_{t} \quad &\xi_{t} \sim \mathcal{N}(0, \sigma^2_{\xi})\\
        \nu_{t+1} &= \nu_{t} + \zeta_{t} \quad &\zeta_{t} \sim \mathcal{N}(0, \sigma^2_{\zeta})\\
        \gamma_{t+1} &= -\sum_{j=1}^{s-1} \gamma_{t+1-j} + \omega_{t} \quad & \omega_{t} \sim \mathcal{N}(0, \sigma^2_{\omega})\\
    \end{aligned}
\end{gather*}
```

#### References
 * Durbin, James, & Siem Jan Koopman. (2012). "Time Series Analysis by State Space Methods: Second Edition." Oxford University Press.


### Local Level Model

The local level model is defined by:
```math
\begin{gather*}
    \begin{aligned}
        y_{t} &=  \mu_{t} + \varepsilon_{t} \quad \varepsilon_{t} \sim \mathcal{N}(0, \sigma^2_{\varepsilon})\\
        \mu_{t+1} &= \mu_{t} + \eta_{t} \quad \eta_{t} \sim \mathcal{N}(0, \sigma^2_{\eta})\\
    \end{aligned}
\end{gather*}
```
#### References
 * Durbin, James, & Siem Jan Koopman. (2012). "Time Series Analysis by State Space Methods: Second Edition." Oxford University Press. pp. 9

### Local Linear Trend 
The linear trend model is defined by:
```math
\begin{gather*}
    \begin{aligned}
        y_{t} &=  \mu_{t} + \gamma_{t} + \varepsilon_{t} \quad &\varepsilon_{t} \sim \mathcal{N}(0, \sigma^2_{\varepsilon})\\
        \mu_{t+1} &= \mu_{t} + \nu_{t} + \xi_{t} \quad &\xi_{t} \sim \mathcal{N}(0, \sigma^2_{\xi})\\
        \nu_{t+1} &= \nu_{t} + \zeta_{t} \quad &\zeta_{t} \sim \mathcal{N}(0, \sigma^2_{\zeta})\\
    \end{aligned}
\end{gather*}
```
#### References
 * Durbin, James, & Siem Jan Koopman. (2012). "Time Series Analysis by State Space Methods:
    Second Edition." Oxford University Press. pp. 44

### Implementing a custom model

Users are able to implement any custom user-defined model. For that, it is necessary to modify "model_utils.jl" file. More specificly, it is necessary to modify "create_X" function to be able to create the corresponding high dimension regression matrix and "get_components_indexes" function to assess the correct compenents in the new proposed model.


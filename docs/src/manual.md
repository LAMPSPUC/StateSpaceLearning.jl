# Manual

## Quickstart

While StateSpaceModels.jl offers a rich array of functionalities, diverse models, and flexible interfaces, most users primarily seek to achieve core tasks: fitting a model and generating forecasts. The following example provides a concise introduction to performing these essential operations effectively.

```julia
using StateSpaceLearning

y = randn(100)

# Instantiate Model
model = StructuralModel(y)

# Fit Model
fit!(model)

# Point Forecast
prediction = StateSpaceLearning.forecast(model, 12) #Gets a 12 steps ahead prediction

# Scenarios Path Simulation
simulation = StateSpaceLearning.simulate(model, 12, 1000) #Gets 1000 scenarios path of 12 steps ahead predictions
```
## Models

The package currently supports the implementation of the StructuralModel, which includes capabilities for handling dynamic exogenous coefficients. If you have suggestions for additional models to include, we encourage you to contribute by opening an issue or submitting a pull request.

```

When using dynamic coefficients:
- The model will create time-varying coefficients for each specified exogenous variable
- Each coefficient will follow the specified cyclical pattern
- When forecasting, you must provide future values for exogenous variables using the `Exogenous_Forecast` parameter

```@docs
StateSpaceLearning.StructuralModel
```

## Fitting

The package currently only allows for the estimation procedure (based on the elastic net) presented in the paper "Time Series Analysis by State Space Learning". We allow some parameters configurations as detailed below. If you have suggestions for additional estimation procedures to include, we encourage you to contribute by opening an issue or submitting a pull request.

```@docs
StateSpaceLearning.fit!
```

## Forecasting and Simulating

The package has functions to make point forecasts multiple steps ahead and to simulate scenarios based on those forecasts. These functions are implemented for both univariate and multivariate cases, with support for exogenous variables and dynamic coefficients.

When using models with exogenous variables:
- For standard exogenous variables, provide future values using the `Exogenous_Forecast` parameter
- For dynamic coefficients, use the same `Exogenous_Forecast` parameter with values for each exogenous variable

```@docs
StateSpaceLearning.forecast
```
```@docs
StateSpaceLearning.simulate
```

## Datasets
The package includes several datasets designed to demonstrate its functionalities and showcase the models. These datasets are stored as CSV files, and their file paths can be accessed either by their names as shown below. In the examples, we utilize DataFrames.jl and CSV.jl to illustrate how to work with these datasets.
```@docs
StateSpaceLearning.AIR_PASSENGERS
```
```@docs
StateSpaceLearning.ARTIFICIAL_SOLARS
```
```@docs
StateSpaceLearning.HOURLY_M4_EXAMPLE
```

# Examples

In this section we show examples of applications and use cases of the package.

## Nile river annual flow

Here we will follow an example from Durbin & Koopman's book. We will use the [`LocalLevel`](@ref) model applied to the annual flow of the Nile river at the city of Aswan between 1871 and 1970.

```@setup nile
using StateSpaceLearning
using Plots
using Dates
```
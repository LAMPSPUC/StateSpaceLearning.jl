@doc raw"""
    AIR_PASSENGERS

The absolute path for the `AIR_PASSENGERS` dataset stored inside StateSpaceLearning.jl.
This dataset provides monthly totals of a US airline passengers from 1949 to 1960.

# References
 * https://www.stata-press.com/data/r12/ts.html
"""
const AIR_PASSENGERS = joinpath(dirname(@__DIR__()), "datasets", "airpassengers.csv")

@doc raw"""
    ARTIFICIAL_SOLARS

The absolute path for the `ARTIFICIAL_SOLARS` dataset stored inside StateSpaceLearning.jl.
This dataset provides an hourly Multivariate Time Series for 3 artificial solar power plants.

"""
const ARTIFICIAL_SOLARS = joinpath(dirname(@__DIR__()), "datasets", "artificial_solars.csv")

@doc raw"""
    HOURLY_M4_EXAMPLE

The absolute path for the `HOURLY_M4_EXAMPLE` dataset stored inside StateSpaceLearning.jl.
This dataset provides an hourly Time Series from the M4 competition dataset.

# References
 * https://github.com/Mcompetitions/M4-methods
"""
const HOURLY_M4_EXAMPLE = joinpath(dirname(@__DIR__()), "datasets", "m4_hourly_example.csv")

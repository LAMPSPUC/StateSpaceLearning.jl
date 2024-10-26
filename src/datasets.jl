@doc raw"""
    AIR_PASSENGERS

The absolute path for the `AIR_PASSENGERS` dataset stored inside StateSpaceLearning.jl.
This dataset provides monthly totals of a US airline passengers from 1949 to 1960.

See more on [Airline passengers](@ref)

# References
 * https://www.stata-press.com/data/r12/ts.html
"""
const AIR_PASSENGERS = joinpath(dirname(@__DIR__()), "datasets", "airpassengers.csv")

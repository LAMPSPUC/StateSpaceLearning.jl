module StateSpaceLearning

using LinearAlgebra, Statistics, GLMNet, Distributions

abstract type StateSpaceLearningModel end

include("structs.jl")
include("models/structural_model.jl")
include("information_criteria.jl")
include("estimation_procedure.jl")
include("utils.jl")
include("datasets.jl")
include("fit_forecast.jl")

export fit!, forecast, simulate, StructuralModel

end # module StateSpaceLearning

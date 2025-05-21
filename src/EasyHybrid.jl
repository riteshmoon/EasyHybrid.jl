module EasyHybrid
using Flux
using DataFrames
using Chain: @chain
using DataFrameMacros
using AxisKeys
using MLJ: partition
using Random

include("utils/tools.jl")
include("models/models.jl")
include("utils/synthetic_test_data.jl")

end

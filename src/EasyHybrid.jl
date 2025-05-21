module EasyHybrid
using Flux
using DataFrames
using Chain: @chain
using DataFrameMacros
using AxisKeys
using MLJ: partition
using Random

include("tools.jl")
include("Hybrid_models.jl")
include("synthetic_test_data.jl")

end

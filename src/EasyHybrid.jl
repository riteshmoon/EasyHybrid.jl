module EasyHybrid
using Flux
using DataFrames
using Chain: @chain
using DataFrameMacros
using MLUtils
using AxisKeys
using MLJ: partition
using Random
using LuxCore
import LuxCore: LuxCore.setup, LuxCore.AbstractLuxContainerLayer
using ChainRulesCore
using Zygote
using Statistics
using ProgressMeter
using Random

include("utils/tools.jl")
include("models/models.jl")
include("utils/synthetic_test_data.jl")
include("utils/losses.jl")
include("train.jl")

end

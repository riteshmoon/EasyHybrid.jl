"""
    EasyHybrid
    
EasyHybrid is a Julia package for hybrid machine learning models, combining neural networks and traditional statistical methods. It provides tools for data preprocessing, model training, and evaluation, making it easier to build and deploy hybrid models.
"""
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
using Statistics: mean, cor
using ProgressMeter
using Random
using NaNStatistics: nanmean
using JLD2
using StyledStrings
using Printf
using Reexport

@reexport import LuxCore
@reexport using Random

include("macro_hybrid.jl")
include("utils/wrap_tuples.jl")
include("utils/io.jl")
include("utils/tools.jl")
include("models/models.jl")
include("utils/synthetic_test_data.jl")
include("utils/logging_loss.jl")
include("utils/loss_fn.jl")
include("train.jl")

end

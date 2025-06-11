using Pkg
Pkg.activate("projects/RsoilComponents")
Pkg.develop(path=pwd())
Pkg.instantiate()

using Revise
using EasyHybrid
using Lux
using Optimisers
using Random
using LuxCore
using CSV, DataFrames
using EasyHybrid.MLUtils

include("projects/RsoilComponents/data/prec_process_data.jl")

df = dfall[!, Not(:timesteps)]

ds_p_f = to_keyedArray(Float32.(df[!, [:cham_temp, :moisture, :rgpot]])) 

target_names = [:R_soil, :R_root, :R_myc, :R_het]
ds_t = to_keyedArray(Float32.(df[!, target_names])) # targets

NN = Lux.Chain(Dense(2, 15, Lux.relu), Dense(15, 15, Lux.relu), Dense(15, 3));
Rsc = Rs_components(NN, (:rgpot, :moisture), target_names, (:cham_temp,), 2.5f0, 2.5f0, 2.5f0)

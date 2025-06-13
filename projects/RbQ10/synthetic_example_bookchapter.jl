using Pkg
Pkg.activate("projects/RbQ10")
Pkg.develop(path=pwd())
Pkg.instantiate()

using EasyHybrid
using Lux
using Optimisers
using GLMakie
using AlgebraOfGraphics
using Random
using LuxCore
using CSV, DataFrames
using EasyHybrid.MLUtils
using EasyHybrid.AxisKeys
using Zygote
using EasyHybrid.JLD2
# data
df_o = CSV.read("projects/RbQ10/data/Synthetic4BookChap.csv", DataFrame, normalizenames=true, missingstring = "NA", dateformat="yyyy-mm-ddTHH:MM:SSZ")
# some pre-processing
dfall = copy(df_o)

rename!(dfall, :TA => :Temp) 
rename!(dfall, :RECO_syn => :Rh)  # rename as in hybrid model

cols_to_select = [:Temp, :Rh, :SW_POT_sm, :SW_POT_sm_diff]

df = select(dfall, cols_to_select...)  # select only the relevant columns
dropmissing!(df)

ds_keyed = to_keyedArray(Float32.(df)) # predictors + forcing

df_pr = df[:, [:Temp, :SW_POT_sm, :SW_POT_sm_diff]]
rename!(df_pr, :Temp => :Temp_pr) # rename predictors
ds_pr = to_keyedArray(Float32.(df_pr)) # predictors

import Flux
#ds_pr = Float32.(Flux.normalise(ds_pr, dims = 1))

df_t_f = df[:, [:Rh, :Temp]]
ds_t_f = to_keyedArray(Float32.(df_t_f)) # target + forcing

ds = cat(ds_t_f, ds_pr, dims = 1) # combine target and predictors

# Define neural network
NN = Lux.Chain(Dense(3, 15, Lux.relu), Dense(15, 15, Lux.relu), Dense(15, 1));
# instantiate Hybrid Model
RbQ10 = RespirationRbQ10(NN, (:SW_POT_sm, :SW_POT_sm_diff, :Temp), (:Rh, ), (:Temp,), 1.f0) # ? do different initial Q10s
# train model
out = train(RbQ10, ds, (:Q10, ); nepochs=200, batchsize=512, opt=Adam(0.01));

## Plotting results
series(out.ps_history; axis=(; xlabel = "epoch", ylabel=""))
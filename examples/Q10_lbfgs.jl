using EasyHybrid
using Lux
using MLUtils
using Random
using Optimization
using OptimizationOptimisers
using ComponentArrays
using GLMakie
using Random
using LuxCore
using CSV, DataFrames

df = CSV.read("/Users/lalonso/Documents/HybridML/data/Rh_AliceHolt_forcing_filled.csv", DataFrame)

df[!, :Temp] = df[!, :Temp] .- 273.15 # convert to Celsius
df_forcing = filter(:Respiration_heterotrophic => !isnan, df)
# df_forcing = df
ds_k = to_keyedArray(Float32.(df_forcing))
yobs =  ds_k(:Respiration_heterotrophic)'[:,:]

NN = Lux.Chain(Dense(2, 15, Lux.relu), Dense(15, 15, Lux.relu), Dense(15, 1));
#? do different initial Q10s
RbQ10 = RespirationRbQ10(NN, (:Rgpot, :Moist), (:Temp,), 2.5f0) 

data = (ds_k([:Rgpot, :Moist, :Temp]), yobs)

(x_train, y_train), (x_val, y_val) = splitobs(data; at=0.8, shuffle=false)
train_loader = DataLoader((x_train, y_train), batchsize=512, shuffle=true);
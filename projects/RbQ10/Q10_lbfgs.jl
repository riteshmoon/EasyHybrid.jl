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
using Statistics
using Printf

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
dataloader = DataLoader((x_train, y_train), batchsize=512, shuffle=true);

ps, st = LuxCore.setup(Random.default_rng(), RbQ10)

ps_ca = ComponentArray(ps)
smodel = StatefulLuxLayer{false}(RbQ10, nothing, st)
# deal with the `Rb` state also here, (; Rb, st), since this is the output from LuxCore.apply.
# ! note that for now is set to `{false}`.

function callback(state, l)
    state.iter % 2 == 1 && @printf "Iteration: %5d, Loss: %.6f\n" state.iter l
    return l < 0.2 ## Terminate if loss is smaller than
end

function lossfn_optim(ps_ca, data)
    ds, y = data
    # ! unpack nan indices here as well
    ŷ, _ = smodel(ds, ps_ca)
    return Statistics.mean(abs2, ŷ .- y)
end

lossfn_optim(ps_ca, (ds_k, yobs))

opt_func = OptimizationFunction(lossfn_optim, Optimization.AutoZygote())
opt_prob = OptimizationProblem(opt_func, ps_ca, dataloader)

epochs = 25
res_adam = solve(opt_prob, Optimisers.Adam(0.001); callback, epochs)

# res_shopia = solve(opt_prob, Optimization.Sophia(); callback, maxiters=epochs)

# ! finetune a bit with L-BFGS
# ?  LBFGS needs to this `convert(Float64, res_adam.u)` which it fails!
# ! but there is more, see issue: https://github.com/LuxDL/Lux.jl/issues/1260

# using ForwardDiff
# opt_func = OptimizationFunction(lossfn_optim, Optimization.AutoForwardDiff())
# opt_prob2 = remake(opt_prob, u0=res_adam.u)
opt_prob = OptimizationProblem(opt_func, res_adam.u, dataloader)
res_lbfgs = solve(opt_prob, Optimization.LBFGS(); callback, maxiters=epochs)
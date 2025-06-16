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
df_o = CSV.read("projects/RbQ10/data/Synthetic4BookChap.csv", DataFrame, normalizenames=true, missingstring = "NA", dateformat="yyyy-mm-ddTHH:MM:SSZ") # /Net/Groups/BGI/scratch/bahrens/DataBookchapter
# some pre-processing
dfall = copy(df_o)

rename!(dfall, :TA => :Temp) 
rename!(dfall, :RECO_syn => :Rh)  # rename as in hybrid model

cols_to_select = [:Temp, :Rh, :SW_POT_sm, :SW_POT_sm_diff]

df = select(dfall, cols_to_select...)  # select only the relevant columns
dropmissing!(df)

ds_keyed = to_keyedArray(Float32.(df)) # predictors + forcing

df_pr = df[:, [:Temp, :SW_POT_sm, :SW_POT_sm_diff]]
#rename!(df_pr, :Temp => :Temp_pr) # rename predictors
ds_pr = to_keyedArray(Float32.(df_pr)) # predictors

# TODO check effect of normailization => I had the feeling this was performing worse in retrieving the synthetic Q10. My intuition is that without normalization, we first have to train the Q10 and the NN has to first get to the right order of magnitude which maybe beneficial
#import Flux
#ds_pr = Float32.(Flux.normalise(ds_pr, dims = 1))

df_t_f = df[:, [:Rh]] # target + forcing                                                                                        
ds_t_f = to_keyedArray(Float32.(df_t_f)) # target + forcing

ds = cat(ds_t_f, ds_pr, dims = 1) # combine target and predictors
axiskeys(ds)
# Define neural network
NN = Lux.Chain(Dense(3, 15, Lux.relu), Dense(15, 15, Lux.relu), Dense(15, 1));
# instantiate Hybrid Model
RbQ10 = RespirationRbQ10(NN, (:SW_POT_sm, :SW_POT_sm_diff, :Temp), (:Rh, ), (:Temp,), 2.5f0) # ? do different initial Q10s
# train model
out = train(RbQ10, ds, (:Q10, ); nepochs=200, batchsize=512, opt=Adam(0.01));

## Plotting results
series(out.ps_history; axis=(; xlabel = "epoch", ylabel=""))

# Test with LBFGS - I had good success with the hybrid example with that optimiser before.
using EasyHybrid.MLUtils

# the Tuple `ds_p, ds_t` is later used for batching in the `dataloader`.
ps, st = LuxCore.setup(Random.default_rng(), RbQ10)
ds = ds .|> Float64
ds_p_f, ds_t = EasyHybrid.prepare_data(RbQ10, ds)
ds_t_nan = .!isnan.(ds_t)

# wrap loss function to get arguments as required by Optimization.jl
ls2 = (p, data) -> lossfn(RbQ10, ds_p_f, (ds_t, ds_t_nan), p, st)

# convert to Float64 for optimization
ps_ca = ComponentArray(ps) .|> Float64
ls2(ps_ca, ds)

# Define optimization problem
opt_func = OptimizationFunction(ls2, Optimization.AutoZygote())
opt_prob = OptimizationProblem(opt_func, ps_ca, ds)

# TODO check if we should do minibatching here
epochs = 10
n_minibatches = 1#length(ds)
Q10s = Float64[]
function callback(state, l)
    push!(Q10s, state.u.Q10[1])
    state.iter % n_minibatches == 0 && @printf "Epoch: %5d, Loss: %.6e\n" state.iter/n_minibatches l
    return l < 1e-8 ## Terminate if loss is small
end

res_adam = solve(opt_prob, Optimisers.Adam(0.01); callback, epochs)
ls2(res_adam.u, ds)

opt_prob = remake(opt_prob; u0=res_adam.u)

# check what maxiters is - is it equivalent to epoch?
res_lbfgs = solve(opt_prob, Optimization.LBFGS(); callback, maxiters=200)
ls2(res_lbfgs.u, ds)

import Plots as pl
pl.plot(Q10s; axis=(; xlabel = "epoch", ylabel="Q10"), title="Q10 during training")

res_lbfgs.u.Q10

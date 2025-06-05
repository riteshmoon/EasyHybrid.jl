using EasyHybrid
using Lux
using Optimisers
using GLMakie
using Random
using LuxCore
using CSV, DataFrames
using EasyHybrid.MLUtils
using EasyHybrid.AxisKeys
using Zygote
# data
df_o = CSV.read("/Users/lalonso/Documents/HybridML/data/Rh_AliceHolt_forcing_filled.csv", DataFrame)

df = copy(df_o)
df[!, :Temp] = df[!, :Temp] .- 273.15 # convert to Celsius
# df = filter(:Respiration_heterotrophic => !isnan, df)
rename!(df, :Respiration_heterotrophic => :Rh)  # rename as in hybrid model
df_forcing = df

ds_p_f = to_keyedArray(Float32.(df_forcing)) # predictors + forcing
ds_t =  ds_p_f([:Rh]) # do the array so that you conserve the name

NN = Lux.Chain(Dense(2, 15, Lux.relu), Dense(15, 15, Lux.relu), Dense(15, 1));
#? do different initial Q10s
RbQ10 = RespirationRbQ10(NN, (:Rgpot, :Moist), (:Rh, ), (:Temp,), 2.5f0)

# ? test lossfn
ps, st = LuxCore.setup(Random.default_rng(), RbQ10)
# the Tuple `ds_p, ds_t` is later used for batching in the `dataloader`.
ds_t_nan = .!isnan.(ds_t)
ls = lossfn(RbQ10, ds_p_f, (ds_t, ds_t_nan), ps, st, LoggingLoss())

# ? play with :Temp as predictors in NN, temperature sensitivity!
# TODO: variance effect due to LSTM vs NN
out = train(RbQ10, (ds_p_f, ds_t), (:Q10, ); nepochs=100, batchsize=512, opt=Adam(0.01));


with_theme(theme_light()) do 
    fig, ax, plt = lines(out.ps_history, color=:grey15;
        #axis = (xscale = log10, yscale=log10),
        label = "Q10",
        figure = (; size = (800, 400)))
    # ylims!(ax, 0.07, 2.5)
    fig
end

with_theme(theme_light()) do 
    fig = Figure(; size = (1200, 600))
    ax_train = Makie.Axis(fig[1, 1], title = "training")
    ax_val = Makie.Axis(fig[2, 1], title = "validation")
    lines!(ax_train, out.ŷ_train.Rh[:], color=:orangered, label = "prediction")
    lines!(ax_train, out.y_train[:], color=:dodgerblue, label ="observation")
    # validation
    lines!(ax_val, out.ŷ_val.Rh[:], color=:orangered, label = "prediction")
    lines!(ax_val, out.y_val[:], color=:dodgerblue, label ="observation")
    axislegend(; position=:lt)
    Label(fig[0,1], "Observations vs predictions", tellwidth=false)
    fig
end

with_theme(theme_light()) do 
    fig = Figure(; size = (1200, 300))
    ax = Makie.Axis(fig[1,1], title = "Loss",
        yscale=log10, xscale=log10
        )
    lines!(ax, out.train_history.sum, color=:orangered, label = "train")
    lines!(ax, out.val_history.sum, color=:dodgerblue, label ="validation")
    # limits!(ax, 1, 1000, 0.04, 1)
    axislegend()
    fig
end


yobs_all =  ds_p_f(:Rh)

ŷ, RbQ10_st = LuxCore.apply(RbQ10, ds_p_f, out.ps, out.st)

with_theme(theme_light()) do 
    fig = Figure(; size = (1200, 600))
    ax_train = Makie.Axis(fig[1, 1], title = "full time series")
    lines!(ax_train, ŷ.Rh[:], color=:orangered, label = "prediction")
    lines!(ax_train, yobs_all[:], color=:dodgerblue, label ="observation")
    axislegend(ax_train; position=:lt)
    Label(fig[0,1], "Observations vs predictions", tellwidth=false)
    fig
end

# ? Rb
lines(out.αst_train.Rb[:])
lines!(ds_p_f(:Moist)[:])
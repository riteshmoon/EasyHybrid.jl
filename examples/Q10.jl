using EasyHybrid
using Lux
using Optimisers
using GLMakie
using Random
using LuxCore
using CSV, DataFrames
using EasyHybrid.MLUtils

# data
df = CSV.read("/Users/lalonso/Documents/HybridML/data/Rh_AliceHolt_forcing_filled.csv", DataFrame)

df[!, :Temp] = df[!, :Temp] .- 273.15 # convert to Celsius
df_forcing = filter(:Respiration_heterotrophic => !isnan, df)
df_forcing = df
ds_k = to_keyedArray(Float32.(df_forcing))
yobs =  ds_k(:Respiration_heterotrophic)'[:,:]

NN = Lux.Chain(Dense(2, 15, Lux.relu), Dense(15, 15, Lux.relu), Dense(15, 1));
#? do different initial Q10s
RbQ10 = RespirationRbQ10(NN, (:Rgpot, :Moist), (:Temp,), 2.5f0) 

# ? play with :Temp as predictors in NN, temperature sensitivity!
# TODO: variance effect due to LSTM vs NN

out = train(RbQ10, (ds_k([:Rgpot, :Moist, :Temp]), yobs), (:Q10, ); nepochs=1000, batchsize=512, opt=Adam(0.01));


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
    ax_train = Axis(fig[1, 1], title = "training")
    ax_val = Axis(fig[2, 1], title = "validation")
    lines!(ax_train, out.ŷ_train[:], color=:orangered, label = "prediction")
    lines!(ax_train, out.y_train[:], color=:dodgerblue, label ="observation")
    # validation
    lines!(ax_val, out.ŷ_val[:], color=:orangered, label = "prediction")
    lines!(ax_val, out.y_val[:], color=:dodgerblue, label ="observation")
    axislegend(; position=:lt)
    Label(fig[0,1], "Observations vs predictions", tellwidth=false)
    fig
end

with_theme(theme_light()) do 
    fig = Figure(; size = (1200, 300))
    ax = Axis(fig[1,1], title = "Loss",
        yscale=log10, xscale=log10
        )
    lines!(ax, out.train_history, color=:orangered, label = "train")
    lines!(ax, out.val_history, color=:dodgerblue, label ="validation")
    # limits!(ax, 1, 1000, 0.04, 1)
    axislegend()
    fig
end


ds_k = to_keyedArray(Float32.(df))
yobs_all =  ds_k(:Respiration_heterotrophic)'[:,:]

ŷ, RbQ10_st = LuxCore.apply(RbQ10, ds_k, out.ps, out.st)

with_theme(theme_light()) do 
    fig = Figure(; size = (1200, 600))
    ax_train = Axis(fig[1, 1], title = "full time series")
    lines!(ax_train, ŷ[:], color=:orangered, label = "prediction")
    lines!(ax_train, yobs_all[:], color=:dodgerblue, label ="observation")
    axislegend(ax_train; position=:lt)
    Label(fig[0,1], "Observations vs predictions", tellwidth=false)
    fig
end


# ? Rb
lines(out.αst_train.Rb[:])
lines!(ds_k(:Moist)[:])
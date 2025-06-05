using Pkg
Pkg.activate("projects/BulkDSOC")
Pkg.develop(path=pwd())
Pkg.instantiate()

using Revise
using EasyHybrid
using Lux
using Optimisers
using GLMakie
using Random
using LuxCore
using CSV, DataFrames
using EasyHybrid.MLUtils
using Statistics
import Plots as pl
# using StatsBase

# ? move the `csv` file into the `BulkDSOC/data` folder (create folder)
df = CSV.read(joinpath(@__DIR__, "./data/lucas_preprocessed.csv"), DataFrame, normalizenames=true)
println(size(df))
df_d = dropmissing(df)
println(size(df_d))

target_names = [:BD, :SOCconc, :CF, :SOCdensity]


names_cov = Symbol.(names(df_d))[4:end-1]
#names_cov = Symbol.(names(df_d))[end-40:end-1]
ds_all = to_keyedArray(df_d);


ds_p = ds_all(names_cov);
ds_t =  ds_all(target_names)

length(ds_t) * 10 # number of samples as guidance for number of parameters
nfeatures = length(names_cov)
p_dropout = 0.2
NN = Lux.Chain(
    Dense(nfeatures, 256, Lux.sigmoid),
    Dropout(p_dropout),
    Dense(256, 128, Lux.sigmoid),
    Dropout(p_dropout),
    Dense(128, 64, Lux.sigmoid),
    Dropout(p_dropout),
    Dense(64, 32, Lux.sigmoid),
    Dropout(p_dropout),
    Dense(32, 3, Lux.sigmoid) # Output layer
)
# ? we might need to set output bounds for the expected parameter values

# ? do different initial oBDs
BulkDSOC = BulkDensitySOC(NN, names_cov, target_names, 1.f0)

ps, st = LuxCore.setup(Random.default_rng(), BulkDSOC)
# the Tuple `ds_p, ds_t` is later used for batching in the `dataloader`.
ds_t_nan = .!isnan.(ds_t)

ls = lossfn(BulkDSOC, ds_p, (ds_t, ds_t_nan), ps, st) # #TODO runs up to here

println(length(names_cov))
out = train(BulkDSOC, (ds_p, ds_t), (:oBD, ); nepochs=100, batchsize=32, opt=AdaMax(0.01));

# plot trained bulk density function
trained_oBD = out[:ŷ_train][:oBD]
trained_mBD = out[:ŷ_train][:mBD]

SOCrange = range(0.0,0.5; step = 0.01)
median_BDc = compute_bulk_density(SOCrange, trained_oBD, median(trained_mBD))
q25_BDc = compute_bulk_density(SOCrange, trained_oBD, quantile(trained_mBD, 0.25))
q75_BDc = compute_bulk_density(SOCrange, trained_oBD, quantile(trained_mBD, 0.75))

pl.plot(ds_t(:SOCconc), ds_t(:BD), seriestype = :scatter, ylabel = :BD, xlabel = :SOCconc)
pl.plot!(collect(SOCrange), median_BDc, width = 4.0, label = "q50")
pl.plot!(collect(SOCrange), q25_BDc, width = 4.0, label = "q25")
pl.plot!(collect(SOCrange), q75_BDc, width = 4.0, label = "q75")

pl.scatter(out[:y_train](:SOCconc),out[:ŷ_train][:SOCconc])
pl.scatter(out[:y_train](:BD),out[:ŷ_train][:BD])
pl.scatter(out[:y_train](:CF),out[:ŷ_train][:CF])
pl.scatter(out[:y_train](:SOCdensity),out[:ŷ_train][:SOCdensity])


# ? analysis, this should also work now!

with_theme(theme_light()) do 
    fig, ax, plt = lines(out.ps_history, color=:grey15;
        #axis = (xscale = log10, yscale=log10),
        label = "oBD",
        figure = (; size = (800, 400)))
    # ylims!(ax, 0.07, 2.5)
    fig
end

with_theme(theme_light()) do 
    fig = Figure(; size = (1200, 800))
    axs = [Makie.Axis(fig[j, 1]) for j in eachindex(target_names)]
    for (i, t_name) in enumerate(target_names)
        lines!(axs[i], getproperty(out.ŷ_train, t_name), color=:orangered, label = "prediction")
        lines!(axs[i], out.y_train(t_name), color=:grey25, label ="observation")
        axs[i].title = "$(t_name)"
    end
    Legend(fig[2,1, Top()], axs[2], nbanks=2, tellwidth=false, halign=0)
    Label(fig[0,1], "Training\nobservations vs predictions", tellwidth=false)
    # do also validation
    axs_v = [Makie.Axis(fig[j, 2]) for j in eachindex(target_names)]
    for (i, t_name) in enumerate(target_names)
        lines!(axs_v[i], getproperty(out.ŷ_val, t_name), color=:orangered, label = "prediction")
        lines!(axs_v[i], out.y_val(t_name), color=:grey25, label ="observation")
        axs_v[i].title = "$(t_name)"
    end
    Legend(fig[2,2, Top()], axs[2], nbanks=2, tellwidth=false, halign=0)
    Label(fig[0,2], "Validation\nobservations vs predictions", tellwidth=false)
    fig
end

with_theme(theme_light()) do 
    fig = Figure(; size = (1200, 300))
    ax = Makie.Axis(fig[1,1], title = "Loss",
        # yscale=log10, 
        xscale=log10
        )
    lines!(ax, out.train_history.sum, color=:orangered, label = "train")
    lines!(ax, out.val_history.sum, color=:dodgerblue, label ="validation")
    # limits!(ax, 1, 1000, 0.04, 1)
    axislegend()
    fig
end
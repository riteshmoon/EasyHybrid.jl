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
# using Plots
# using StatsBase

# ? move the `csv` file into the `BulkDSOC/data` folder (create folder)
df = CSV.read(joinpath(@__DIR__, "./data/lucas_preprocessed.csv"), DataFrame, normalizenames=true)
println(size(df))
df_d = dropmissing(df)
println(size(df_d))

target_names = [:BD, :SOCconc, :CF, :SOCdensity]
names_cov = Symbol.(names(df_d))[4:end-1]
ds_all = to_keyedArray(df_d);

ds_p = ds_all(names_cov);
ds_t =  ds_all(target_names)

nfeatures = length(names_cov)
NN = Lux.Chain(Dense(nfeatures, nfeatures*2, Lux.relu), Dense(nfeatures*2, nfeatures*2, Lux.relu),Dense(nfeatures*2, 15, Lux.relu) ,Dense(15, 3))
# ? we might need to set output bounds for the expected parameter values

# ? do different initial oBDs
BulkDSOC = BulkDensitySOC(NN, names_cov, target_names, 0.3f0)

ps, st = LuxCore.setup(Random.default_rng(), BulkDSOC)
# the Tuple `ds_p, ds_t` is later used for batching in the `dataloader`.
ds_t_nan = .!isnan.(ds_t)

ls = lossfn(BulkDSOC, ds_p, (ds_t, ds_t_nan), ps, st) # #TODO runs up to here

println(length(names_cov))
out = train(BulkDSOC, (ds_p, ds_t), (:oBD, ); nepochs=100, batchsize=512, opt=Adam(0.001));

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
        yscale=log10, xscale=log10
        )
    lines!(ax, out.train_history, color=:orangered, label = "train")
    lines!(ax, out.val_history, color=:dodgerblue, label ="validation")
    # limits!(ax, 1, 1000, 0.04, 1)
    axislegend()
    fig
end
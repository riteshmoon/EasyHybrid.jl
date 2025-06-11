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
using AlgebraOfGraphics
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

# plot train history
series(out.train_history)
# custom plot
with_theme(theme_ggplot2()) do 
   series(out.train_history; color=resample_cmap(:viridis, 5),
    axislegend = (; nbanks=5, framewidth=0.1, backgroundcolor=:white),
    axis = (; xscale=log10, yscale=log10, xlabel="epoch", title = "Losses"),
    figure = (; size = (650, 400)))
end

yvars = target_names # [:BD, :SOCconc, :CF, :SOCdensity]
xvars = Symbol.(string.(yvars) .* "_pred")

layers = visual(Scatter, alpha = 0.35)
plt = data(out.train_obs_pred) * layers * mapping(xvars, yvars, col=dims(1) => renamer(string.(yvars)))
plt *= mapping(color = dims(1) => renamer(string.(xvars))=> "Metrics")
# linear
l_linear = linear() * visual(color=:grey25)
plt += data(out.train_obs_pred) * l_linear *  mapping(xvars, yvars, col=dims(1) => renamer(string.(yvars)))

summary_stats = Dict(string.(target_names) .=> rand(4))

with_theme(theme_light()) do 
   draw(plt, scales(
        X = (; label = rich("Prediction", font=:bold)),
        Y = (; label = "Observation"),
        Color = (; palette = [:tomato, :teal, :orange, :dodgerblue3])
   ),
    # legend = (; show = false),
    legend = (; position=:bottom, titleposition=:left, merge=false),
    facet = (; linkxaxes = :none, linkyaxes = :none,),
    figure = (; size = (1400, 400))
) 
end


layers = visual(Scatter, alpha=0.65)
plt = data(out.val_obs_pred) * layers * mapping(:index, [:BD, :BD_pred]) *
    mapping(color=dims(1) => renamer(string.([:BD, :BD_pred])))

with_theme(theme_light()) do
    colors = ["a" => :tomato, "c" => :lime,]
    draw(plt, scales(Color = (; palette = [:grey25, :tomato])),
        legend =(position=:top,),
        figure = (; size = (1400, 400)))    
end
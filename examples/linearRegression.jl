using EasyHybrid
using Lux
using Optimisers
using GLMakie
using Random
using LuxCore

ds_k = gen_linear_data(; seed=123)
yobs =  ds_k(:obs)'[:,:]

NN = Lux.Chain(Dense(2, 15, Lux.relu), Dense(15, 15, Lux.relu), Dense(15, 1));
lhm = LinearHM(NN, (:x1, :x2), (:x3,), 0.0f0)

# ps_nn, st_nn = LuxCore.setup(Random.default_rng(), NN)
ps, st = LuxCore.setup(Random.default_rng(), lhm)
# opt_state = Optimisers.setup(Adam(), ps)

out = train(lhm, (ds_k([:x1, :x2, :x3]), yobs), (:β, ); nepochs=10_000, batchsize=100, opt=Adam(0.01));

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
    fig, ax, plt = lines(out.ps_history, color=:grey15;
        axis = (xscale = log10, yscale=log10), label = "β",
        figure = (; size = (800, 400)))
    ylims!(ax, 0.07, 2.5)
    fig
end
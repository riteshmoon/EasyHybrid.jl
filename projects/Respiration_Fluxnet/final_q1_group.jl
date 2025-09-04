# CC BY-SA 4.0
# =============================================================================
# Setup and Data Loading
# =============================================================================
using Pkg
project_path = "projects/Respiration_Fluxnet"
Pkg.activate(project_path)
Pkg.develop(path=pwd())
#Pkg.instantiate()

# start using the package
using EasyHybrid
using AxisKeys
using WGLMakie

include("Data/load_data.jl")

# =============================================================================
# Load data
# =============================================================================

# copy data to data/data20240123/ from here /Net/Groups/BGI/work_4/scratch/jnelson/4Sinikka/data20240123
# or adjust the path to /Net/Groups/BGI/work_4/scratch/jnelson/4Sinikka/data20240123 + FluxNetSite

site = "US-SRG"

fluxnet_data = load_fluxnet_nc(joinpath("/ptmp/mp002/ellis/DataEasyHybrid/data20240123", "$site.nc"), timevar="date")

# explore data structure
println(names(fluxnet_data.timeseries))
println(fluxnet_data.scalars)
println(names(fluxnet_data.profiles))

# select timeseries data
df = fluxnet_data.timeseries

# =============================================================================
# Mechanistic Model Definition
# =============================================================================
function flux_part_mechanistic_model(;SW_IN, TA, RUE, Rb, Q10)
    # -------------------------------------------------------------------------
    # Arguments:
    #   SW_IN     : Incoming shortwave radiation
    #   TA      : Air temperature
    #   RUE     : Radiation Use Efficiency
    #   Rb      : Basal respiration
    #   Q10     : Temperature sensitivity 
    #
    # Returns:
    #   NEE     : Net Ecosystem Exchange
    #   RECO    : Ecosystem respiration
    #   GPP     : Gross Primary Production
    # -------------------------------------------------------------------------

    # Calculate fluxes
    GPP = SW_IN .* RUE ./ 12.011f0  # µmol/m²/s
    RECO = Rb .* Q10 .^ (0.1f0 .* (TA .- 15.0f0))
    NEE = RECO .- GPP
    
    return (;NEE, RECO, GPP, Q10, RUE)
end

# =============================================================================
# Parameter container for the mechanistic model
# =============================================================================

# Define parameter structure with bounds
parameters = (
    #            default                  lower                     upper                description
    RUE      = ( 0.1f0,                  0.0f0,                   1.0f0 ),            # Radiation Use Efficiency [g/MJ]
    Rb       = ( 1.0f0,                  0.0f0,                   6.0f0 ),            # Basal respiration [μmol/m²/s]
    Q10      = ( 1.5f0,                  1.0f0,                   4.0f0 ),            # Temperature sensitivity factor [-]
)

# =============================================================================
# Hybrid Model Creation
# =============================================================================
target_FluxPartModel = [:NEE]
forcing_FluxPartModel = [:SW_IN, :TA]

predictors = (Rb = [:SWC_shallow, :P, :WS, :cos_dayofyear, :sine_dayofyear], 
              RUE = [:TA, :P, :WS, :SWC_shallow, :VPD, :SW_IN_POT, :dSW_IN_POT, :dSW_IN_POT_DAY])

global_param_names = [:Q10]
hybrid_model = constructHybridModel(
    predictors,
    forcing_FluxPartModel,
    target_FluxPartModel,
    flux_part_mechanistic_model,
    parameters,
    global_param_names,
    scale_nn_outputs=true,
    hidden_layers = [32, 32],
    activation = sigmoid,
    input_batchnorm = true,
    start_from_default = false
)

# =============================================================================
# Model Training
# =============================================================================
using OhMyThreads: @tasks

# Set the data directory path
data_dir = "/ptmp/mp002/ellis/DataEasyHybrid/data20240123"

# Get full paths to all *.nc files
nc_files = filter(f -> endswith(f, ".nc") && isfile(f),
                  readdir(data_dir; join=true))

# Extract site names (filename without extension)
sites = first.(splitext.(basename.(nc_files)))

#using Base.Threads
# Tropical sites: (no T seasonality)
#     BR-CST (Brazil - Caatinga Serra Talhada)
#     MY-PSO (Malaysia - Pasoh National Park)
#     GH-Ank (Ghana - Ankasa)
# Temperate sites: (strong T seasonality)
#     DE-Hai (Germany - Hainich)
#     US-Ho2 (US - Howland Forest)
#     CN-Cha (China-Changbaishan)
#Pkg.add("CSV")
using CairoMakie
using CSV
@tasks for site in ["BR-CST", "MY-PSO", "GH-Ank", "DE-Hai", "US-HWB", "CN-Cha"] # sites[randperm(length(sites))[1:3]]
    fluxnet_data = load_fluxnet_nc(joinpath("/ptmp/mp002/ellis/DataEasyHybrid/data20240123", "$site.nc"), timevar="date")
    df = fluxnet_data.timeseries

    # only good quality data: a record is a measured value (*_QC=0), or the quality level of the gap-filling that was used for that record (*_QC=1 better, *_QC=3 worse quality)
    df = df[df.NEE_QC .== 0, :]

    # Check number of non-NaN values in predictors, forcing, and target
    predictor_cols = unique(vcat(values(predictors)...))
    forcing_cols = forcing_FluxPartModel
    target_cols = target_FluxPartModel

    println("Site: $site")
    train_on_site = true
    for col in vcat(predictor_cols, forcing_cols, target_cols)
        n_nonan = sum(.!ismissing.(df[!, col]))
        if n_nonan < 1000
            println("Column $col has only $n_nonan non-NaN values")
            train_on_site = false
        end
    end

    if train_on_site
        out_Generic = train(
            hybrid_model, df, ();
            nepochs = 100,
            batchsize = 512,
            opt = RMSProp(0.01),
            loss_types = [:nse, :mse],
            training_loss = :nse,
            random_seed = 123,
            yscale = identity,
            monitor_names = [:RUE, :Q10],
            patience = 20,
            shuffleobs = true,
            plotting = false,
            show_progress = false,
            hybrid_name = "",
            folder_to_save = "_$site"
        )
    end

    output_file = joinpath(
        "/raven/u/mp182/EasyHybrid_work/projects/Respiration_Fluxnet/",
        "output_tmp_$site/best_model.jld2"
    )
    all_groups = get_all_groups(output_file)
    psst, _ = load_group(output_file, :HybridModel_MultiNNHybridModel)
    ps_learned, st_learned = psst[end][1], psst[end][2]

    # forward model Run
    forward_run = hybrid_model(df, ps_learned, st_learned)
    forward_run.NEE_pred
    forward_run.GPP_pred
    forward_run.RECO_pred
    forward_run.RUE_pred

    CSV.write(
        joinpath(
            "/raven/u/mp182/EasyHybrid_work/projects/Respiration_Fluxnet/output_tmp_$site/",
            "data.csv"
        ),
        forward_run
    )
end

# -- Plot --
using TidierPlots
using WGLMakie
@tasks for site in ["BR-CST", "MY-PSO", "GH-Ank", "DE-Hai", "US-HWB", "CN-Cha"]
    path = "/raven/u/mp182/EasyHybrid_work/projects/Respiration_Fluxnet/output_tmp_$site/"
    df2 = CSV.read(
        joinpath(
            "/raven/u/mp182/EasyHybrid_work/projects/Respiration_Fluxnet/output_tmp_$site/",
            "data.csv"
        ),
        DataFrame
    )
    beautiful_makie_theme = Attributes(fonts=(;regular="CMU Serif"))

    # ----- GPP NIGHT ---
    fig = Figure(; size = (1200, 600), fontsize = 24)
    ax = CairoMakie.Axis(fig[1,1], title="GPP Nighttime", xlabel="time", ylabel="GPP")
    lines!(ax, df2.time, df2.GPP_NT, color=:grey25, label="Nighttime GPP")
    lines!(ax, df2.time, df2.GPP_pred, color=:tomato, label="Hybrid GPP")
    axislegend(ax; position=:rt)  # right-top corner by default
    save(joinpath(path, "fig_GPP_NT.png"), fig)   

    # ---- GPP DAY ---
    fig = Figure(; size = (1200, 600), fontsize = 24)
    ax = CairoMakie.Axis(fig[1,1], title="GPP Daytime", xlabel="time", ylabel="GPP")
    lines!(ax, df2.time, df2.GPP_DT, color=:grey25, label="Daytime GPP")
    lines!(ax, df2.time, df2.GPP_pred, color=:tomato, label="Hybrid GPP")
    axislegend(ax; position=:rt)  # right-top corner by default
    save(joinpath(path, "fig_GPP_DT.png"), fig)

    # idx = .!isnan.(df2.GPP_DT) .& .!isnan.(df2.GPP_pred)
    # EasyHybrid.poplot(
    #     df2.GPP_DT[idx],
    #     df2.GPP_pred[idx],
    #     "GPP",
    #     xlabel = "Daytime GPP",
    #     ylabel = "Hybrid GPP"
    # )
    # save(joinpath(path, "fig_GPP_DT.png"), current_figure())

    # ---- RECO Night ---
    fig = Figure(; size = (1200, 600), fontsize = 24)
    ax = CairoMakie.Axis(fig[1,1], title="RECO Nighttime", xlabel="time", ylabel="RECO")
    lines!(ax, df2.time, df2.RECO_NT, color=:grey25, label="Nighttime RECO")
    lines!(ax, df2.time, df2.RECO_pred, color=:tomato, label="Hybrid RECO")
    axislegend(ax; position=:rt)  # right-top corner by default
    save(joinpath(path, "fig_RECO_NT.png"), fig)

    # idx = .!isnan.(df2.RECO_DT) .& .!isnan.(df2.RECO_pred)
    # EasyHybrid.poplot(
    #     df2.RECO_DT[idx],
    #     df2.RECO_pred[idx],
    #     "RECO",
    #     xlabel = "Night RECO",
    #     ylabel = "Hybrid RECO"
    # )
    # save(joinpath(path, "fig_RECO_NT.png"), current_figure())

    # ---- RECO Day ---
    fig = Figure(; size = (1200, 600), fontsize = 24)
    ax = CairoMakie.Axis(fig[1,1], title="RECO Daytime", xlabel="time", ylabel="RECO")
    lines!(ax, df2.time, df2.RECO_DT, color=:grey25, label="Daytime RECO")
    lines!(ax, df2.time, df2.RECO_pred, color=:tomato, label="Hybrid RECO")
    axislegend(ax; position=:rt)  # right-top corner by default
    save(joinpath(path, "fig_RECO_DT.png"), fig)

    # idx = .!isnan.(df2.RECO_DT) .& .!isnan.(df2.RECO_pred)
    # EasyHybrid.poplot(
    #     df2.RECO_DT[idx],
    #     df2.RECO_pred[idx],
    #     "RECO",
    #     xlabel = "Daytime RECO",
    #     ylabel = "Hybrid RECO"
    # )
    # save(joinpath(path, "fig_RECO_DT.png"), current_figure())
end

df2 = CSV.read(
    joinpath(
        "/raven/u/mp182/EasyHybrid_work/projects/Respiration_Fluxnet/output_tmp_BR-CST/",
        "data.csv"
    ),
    DataFrame
)

df2.time
df2.GPP_pred
df2.GPP_NT

fig = Figure(; size = (1200, 600), fontsize = 24)
ax = CairoMakie.Axis(fig[1,1], title="GPP_n_time", xlabel="time", ylabel="GPP")
lines!(ax, df2.time, df2.GPP_NT, color=:grey25, label="GPP_NT")
lines!(ax, df2.time, df2.GPP_DT, color=:purple, label="GPP_DT")
lines!(ax, df2.time, df2.GPP_pred, color=:tomato, label="GPP_pred")
axislegend(ax; position=:rt)  # right-top corner by default
fig
save("BR-CST.png", fig)


beautiful_makie_theme = Attributes(fonts=(;regular="CMU Serif"))
ggplot(df2) + geom_line(aes(x=:time, y=:GPP_pred), alpha=0.75) +
geom_line(aes(x=:time, y=:GPP_NT))# +beautiful_makie_theme

ggplot(df2, @aes(x=:time)) +
    geom_line(@aes(y=:GPP_pred), alpha=0.75) +
    geom_line(@aes(y=:GPP_NT))

# # read the trained weights and biases from disk
# site = "US-HWB"
# output_file = joinpath("/raven/u/mp182/EasyHybrid_work/projects/Respiration_Fluxnet/", "output_tmp_$site/best_model.jld2")
# all_groups = get_all_groups(output_file)

# psst, _ = load_group(output_file, :HybridModel_MultiNNHybridModel)

# ps_learned, st_learned = psst[end][1], psst[end][2]

# # forward model Run
# forward_run = hybrid_model(df, ps_learned, st_learned)

# forward_run.NEE_pred
# forward_run.GPP_pred
# forward_run.RECO_pred
# forward_run.RUE_pred

# # https://nrennie.rbind.io/blog/introduction-julia-r-users/
# using TidierPlots
# using WGLMakie
# beautiful_makie_theme = Attributes(fonts=(;regular="CMU Serif"))

# ggplot(forward_run, @aes(x=GPP_NT, y=GPP_pred)) + geom_point() + beautiful_makie_theme

# idx = .!isnan.(forward_run.GPP_NT) .& .!isnan.(forward_run.GPP_pred)
# EasyHybrid.poplot(forward_run.GPP_NT[idx], forward_run.GPP_pred[idx], "GPP", xlabel = "Nighttime GPP", ylabel = "Hybrid GPP")
# save("fig.png", current_figure())

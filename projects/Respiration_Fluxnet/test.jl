#cd("Z:\\summer_school_jena\\project\\easy_hybrid\\EasyHybrid.jl")
# include("projects/Respiration_Fluxnet/test.jl")

using Pkg
project_path = @__DIR__  # ...\EasyHybrid.jl\projects\Respiration_Fluxnet
Pkg.activate(project_path)
Pkg.instantiate()

using EasyHybrid
using AxisKeys
using WGLMakie
using CSV, DataFrames
##
include(joinpath(@__DIR__, "Data", "load_data.jl"))

ncpath = joinpath(@__DIR__, "Data", "data20240123", "US-SRG.nc")
@assert isfile(ncpath) "Missing file: $(ncpath)"
fluxnet_data = load_fluxnet_nc(ncpath, timevar="date")

println(names(fluxnet_data.timeseries))
println(fluxnet_data.scalars)
println(names(fluxnet_data.profiles))

# select timeseries data
df = fluxnet_data.timeseries

println(names(df))

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
### without sin_dayofyear and cos_dayofyear
predictors = (Rb = [:SWC_shallow, :P, :WS], 
              RUE = [:TA, :P, :WS, :SWC_shallow, :VPD, :SW_IN_POT, :dSW_IN_POT, :dSW_IN_POT_DAY])
## , :sine_dayofyear, :cos_dayofyear was removed to test if it works without them
global_param_names = [:Q10]

hybrid_model = constructHybridModel(
    predictors,
    forcing_FluxPartModel,
    target_FluxPartModel,
    flux_part_mechanistic_model,
    parameters,
    global_param_names,
    scale_nn_outputs=true,
    hidden_layers = [15, 15],
    activation = sigmoid,
    input_batchnorm = true,
    start_from_default = false
)

# =============================================================================
# Model Training
# =============================================================================

out_Generic = train(hybrid_model, df, (); nepochs=5, batchsize=512, opt=RMSProp(0.01), 
                    loss_types=[:nse, :mse], training_loss=:nse, random_seed=123, 
                    yscale = identity, monitor_names=[:RUE, :Q10], patience = 30, 
                    shuffleobs = true)

EasyHybrid.poplot(out_Generic)

q10 = EasyHybrid.scale_single_param(:Q10, out_Generic.ps.Q10, hybrid_model.parameters)[1]
println("Q10 (parameter) = ", q10)

train_q10 = out_Generic.train_diffs.Q10
val_q10   = out_Generic.val_diffs.Q10

## saving this results to a file

CSV.write(joinpath(@__DIR__, "output_tmp", "Q10_results.csv"), DataFrame(q10=[q10]))


### WITH SINE AND COS DAYOFYEAR
predictors = (Rb = [:SWC_shallow, :P, :WS,:sine_dayofyear, :cos_dayofyear], 
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
    hidden_layers = [15, 15],
    activation = sigmoid,
    input_batchnorm = true,
    start_from_default = false
)

# =============================================================================
# Model Training
# =============================================================================

out_Generic = train(hybrid_model, df, (); nepochs=5, batchsize=512, opt=RMSProp(0.01), 
                    loss_types=[:nse, :mse], training_loss=:nse, random_seed=123, 
                    yscale = identity, monitor_names=[:RUE, :Q10], patience = 30, 
                    shuffleobs = true)

EasyHybrid.poplot(out_Generic)

q10 = EasyHybrid.scale_single_param(:Q10, out_Generic.ps.Q10, hybrid_model.parameters)[1]
println("Q10 (parameter) = ", q10)

train_q10 = out_Generic.train_diffs.Q10
val_q10   = out_Generic.val_diffs.Q10

## saving this results to a file

CSV.write(joinpath(@__DIR__, "output_tmp", "Q10_results.csv"), DataFrame(q10=[q10]))


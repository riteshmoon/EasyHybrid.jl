
export BulkDensitySOC

"""
    BulkDensitySOC(NN, predictors, oBD)

A hybrid model with a neural network `NN`, `predictors` and one global parameter oBD.
"""
struct BulkDensitySOC{D, T1, T2} <: LuxCore.AbstractLuxContainerLayer{(:NN, :predictors, :oBD)}
    NN
    predictors
    oBD # organic matter bulk density
    function BulkDensitySOC(NN::D, predictors::T1, oBD::T2) where {D, T1, T2}
        new{D, T1, T2}(NN, collect(predictors), [oBD])
    end
end

# ? oBD is a parameter, so expand the initialparameters!
function LuxCore.initialparameters(::AbstractRNG, layer::BulkDensitySOC)
    ps, _ = LuxCore.setup(Random.default_rng(), layer.NN)
    return (; ps, oBD = layer.oBD,)
end

function LuxCore.initialstates(::AbstractRNG, layer::BulkDensitySOC)
    _, st = LuxCore.setup(Random.default_rng(), layer.NN)
    return (; st)
end

"""
    BulkDensitySOC(NN, predictors, oBD)(ds_k)

# Hybrid model for bulk density based on the Federer (1993) paper http://dx.doi.org/10.1139/x93-131 plus SOC concnetration, density and coarse fraction
"""
function (hm::BulkDensitySOC)(ds_k, ps, st::NamedTuple)
    p = ds_k(hm.predictors)
    
    out, st = LuxCore.apply(hm.NN, p, ps.ps, st.st)

    SOCconc = out[1] #TODO has to be a ratio 
    CF = out[2]
    mBD = out[3] # mineral bulk density

    oF = SOCconc * 1.724 #TODO has to be a ratio

    BD = ps.oBD * mBD / (oF * mBD .+ (1.0f0 - oF) * ps.oBD)
    
    SOCdensity = SOCconc * BD * (1 .- CF)

    return (; SOCconc, CF, BD, SOCdensity), (; mBD, st)
end
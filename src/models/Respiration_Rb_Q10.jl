
export RespirationRbQ10

"""
    RespirationRbQ10(NN, predictors, forcing, Q10)

A linear hybrid model with a neural network `NN`, `predictors` and `forcing` terms.
"""
struct RespirationRbQ10{D, T1, T2, T3} <: LuxCore.AbstractLuxContainerLayer{(:NN, :predictors, :forcing, :Q10)}
    NN
    predictors
    forcing
    Q10
    function RespirationRbQ10(NN::D, predictors::T1, forcing::T2, Q10::T3) where {D, T1, T2, T3}
        new{D, T1, T2, T3}(NN, collect(predictors), collect(forcing), [Q10])
    end
end

# ? Q10 is a parameter, so expand the initialparameters!
function LuxCore.initialparameters(::AbstractRNG, layer::RespirationRbQ10)
    ps, _ = LuxCore.setup(Random.default_rng(), layer.NN)
    return (; ps, Q10 = layer.Q10,)
end

function LuxCore.initialstates(::AbstractRNG, layer::RespirationRbQ10)
    _, st = LuxCore.setup(Random.default_rng(), layer.NN)
    return (; st)
end

"""
    RespirationRbQ10(NN, predictors, forcing, Q10)(ds_k)

# Model definition `ŷ = Rb(αᵢ(t)) * Q10^((T(t) - T_ref)/10)`

ŷ (respiration rate) is computed as a function of the neural network output `Rb(αᵢ(t))` and the temperature `T(t)` adjusted by the reference temperature `T_ref` (default 15°C) using the Q10 temperature sensitivity factor.
````
"""
function (hm::RespirationRbQ10)(ds_k, ps, st::NamedTuple)
    p = ds_k(hm.predictors)
    x = ds_k(hm.forcing)
    
    Rb, st = LuxCore.apply(hm.NN, p, ps.ps, st.st) #! NN(αᵢ(t)) ≡ Rb(T(t), M(t))

    ŷ = Rb .* ps.Q10 .^(0.1f0 * (x .- 15.0f0)) # ? should 15°C be the reference temperature also an input variable?

    return ŷ, (; Rb, st)
end
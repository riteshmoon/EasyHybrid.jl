
export LinearHM

"""
    LinearHM(NN, predictors, forcing, β)

A linear hybrid model with a neural network `NN`, `predictors` and `forcing` terms.
"""
struct LinearHM{D, T1, T2, T3} <: LuxCore.AbstractLuxContainerLayer{(:NN, :predictors, :forcing, :β)}
    NN
    predictors
    forcing
    β
    function LinearHM(NN::D, predictors::T1, forcing::T2, β::T3) where {D, T1, T2, T3}
        new{D, T1, T2, T3}(NN, collect(predictors), collect(forcing), [β])
    end
end

# ? β is a parameter
LuxCore.initialparameters(::AbstractRNG, layer::LinearHM) = (β = layer.β,)
LuxCore.initialstates(::AbstractRNG, layer::LinearHM) = NamedTuple()

# function LuxCore.setup(rng::Random.AbstractRNG, lhm::LinearHM)
#     ps, st = LuxCore.setup(rng, lhm.NN)
#     return (ps, st)
# end

"""
    LinearHM(NN, predictors, forcing, β)(ds_k)

# Model definition `ŷ = α x + β`

Apply the linear hybrid model to the input data `ds_k` (a `KeyedArray` with proper dimensions).
The model uses the neural network `NN` to compute new `α`'s based on the `predictors` and then computes `ŷ` using the `forcing` term `x`.

Returns:

A tuple containing the predicted values `ŷ` and a named tuple with the computed values of `α` and the state `st`.

## Example:
````julia
using Lux
using EasyHybrid
using Random
using AxisKeys

ds_k = KeyedArray(rand(Float32, 3,4); data=[:a, :b, :c], sample=1:4)
m = Lux.Chain(Dense(2, 5), Dense(5, 1))
# Instantiate the model
# Note: The model is initialized with a neural network and the predictors and forcing terms
lh_model = LinearHM(m, (:a, :b), (:c,), 1.5f0)
rng = Random.default_rng()
Random.seed!(rng, 0)
ps, st = LuxCore.setup(rng, lh_model)
# Apply the model to the data
ŷ, αst = LuxCore.apply(lh_model, ds_k, ps, st)
````
"""
function (lhm::LinearHM)(ds_k, ps, st::NamedTuple)
    p = ds_k(lhm.predictors)
    x = ds_k(lhm.forcing)
    α, st = LuxCore.apply(lhm.NN, p, ps, st)
    ŷ = α .* x .+ lhm.β

    return ŷ, (; α, st)
end


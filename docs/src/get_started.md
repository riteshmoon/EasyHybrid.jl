# EasyHybrid.jl

`EasyHybrid.jl` extends `Lux.jl` layers to define custom hybrid models. A complete definition follows the next steps:

## Model struct
Define your model as a sub-type of an `LuxCore.AbstractLuxContainerLayer`. Namely,

```julia
struct YourHybridModelName{D, T1, T2, T3, T4} <: LuxCore.AbstractLuxContainerLayer{(:NN, :predictors, :forcing, :targets, :α)}
  NN
  predictors
  forcing
  targets
  α
  function YourHybridModelName(NN::D, predictors::T1, forcing::T2, targets::T3, α::T4) where {D, T1, T2, T3, T4}
      new{D, T1, T2, T3, T4}(NN, collect(predictors), collect(targets), collect(forcing), [α])
  end
end
```

## Initial parameters and states

::: code-group

```julia [initial parameters]
function LuxCore.initialparameters(::AbstractRNG, layer::YourHybridModelName)
  ps, _ = LuxCore.setup(Random.default_rng(), layer.NN)
  return (; ps, α = layer.α,) # these parameters are trainable! [!code warning]
end
```

```julia [initial states]
function LuxCore.initialstates(::AbstractRNG, layer::RespirationRbQ10)
  _, st = LuxCore.setup(Random.default_rng(), layer.NN)
  return (; st) # none of the possible additional arguments/variables here are trainable! [!code warning]
end
```

:::

## Definition
Model definition, how does the actual model operates!

```julia
function (hm::YourHybridModelName)(ds_k, ps, st::NamedTuple)
  # data selection
  p = ds_k(hm.predictors)
  x = Array(ds_k(hm.forcing))
  
  # output from Neural network application 
  β, st = LuxCore.apply(hm.NN, p, ps.ps, st.st) # [!code highlight]

  # equation!
  ŷ = β .* ps.α .^(0.1f0 * (x .- 15.0f0)) # [!code warning]

  return (; ŷ), (; β, st) # always output predictions and states as two tuples
end
```

## Loss function
Loss? what is your approach for optimization?

```julia
function lossfn(HM::YourHybridModelName, ds_p, (ds_t, ds_t_nan), ps, st)
  targets = HM.targets
  ŷ, _ = HM(ds_p, ps, st)
  y = ds_t(targets)
  y_nan = ds_t_nan(targets)

  loss = 0.0
  for k in targets
      loss += mean(abs2, (ŷ[k][y_nan(k)] .- y(k)[y_nan(k)]))
  end
  return loss
end
```

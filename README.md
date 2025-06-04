# EasyHybrid
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://verbose-chainsaw-z2zjmlp.pages.github.io/dev/)
[![CI](https://github.com/HybridModelling/EasyHybrid/actions/workflows/CI.yml/badge.svg)](https://github.com/HybridModelling/EasyHybrid/actions/workflows/CI.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/HybridModelling/EasyHybrid/blob/main/LICENSE)



> [!CAUTION]
> Work in progress


## Hybrid modelling for teaching purposes

The idea of this repo to provide a relatively simple approach for hybrid modelling, i.e. creating a model which combines machine learning with domain scientific modelling. In a general sense $y = g(f(x), z, \theta)$ where $g$ is a parametric function with parameters $\theta$ to be learned, and $f$ is non-parametric and to be learned. Here $f$ is represented with a neural net in Flux.jl.  

## Start using it

```julia
using EasyHybrid
```
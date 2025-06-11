# EasyHybrid.jl
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://verbose-chainsaw-z2zjmlp.pages.github.io/dev/)
[![CI](https://github.com/EarthyScience/EasyHybrid.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/EarthyScience/EasyHybrid.jl/actions/workflows/CI.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/EarthyScience/EasyHybrid.jl/blob/main/LICENSE)



> [!CAUTION]
> Work in progress


## Hybrid modelling for teaching purposes

The idea of this repo to provide a relatively simple approach for hybrid modelling, i.e. creating a model which combines machine learning with domain scientific modelling. In a general sense $y = g(f(x), z, \theta)$ where $g$ is a parametric function with parameters $\theta$ to be learned, and $f$ is non-parametric and to be learned. Here $f$ is represented with a neural net in Flux.jl.  

## Installation
Clone the repository

```sh
git clone https://github.com/EarthyScience/EasyHybrid.jl.git
```

and start using it by opening one of the `env` in `projects`, i.e. Q10.jl. There executing the first 4 lines should get you all needed dependencies. `shift + enter`.

## If you want to start adding new functionality to it then do

```sh
EasyHybrid $ julia # call julia in the EasyHybrid directory
```

```sh
julia> ] # ']' should be pressed, this is the pkg mode
```

```sh
pkg > activate . # this activates this project
```

install dependencies

```sh
pkg > instantiate
```

and now you are good to go!

```julia
using EasyHybrid
```
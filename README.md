# QuestionnaireFastTransform

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MagineZ.github.io/QuestionnaireFastTransform.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MagineZ.github.io/QuestionnaireFastTransform.jl/dev/)
[![Build Status](https://github.com/MagineZ/QuestionnaireFastTransform.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MagineZ/QuestionnaireFastTransform.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MagineZ/QuestionnaireFastTransform.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MagineZ/QuestionnaireFastTransform.jl)

This code is built in Julia environment and calls Python 'pyquest' package.
We build more features based on the original 'pyqyest' from: https://github.com/gmishne/pyquest

## SETUP


Download and install the package as follows in Julia.

```julia
using Pkg
Pkg.activate("path/to/QuestionnaireFastTransform") # change the path here in your local PC
using QuestionnaireFastTransform
```
Also, make sure `numpy` is installed in the system.

```julia
ENV["PYTHON"] = ""
using Pkg
Pkg.build("PyCall")
using Conda
Conda.add("numpy")


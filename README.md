# QuestionnaireFastTransform

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MagineZ.github.io/QuestionnaireFastTransform.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MagineZ.github.io/QuestionnaireFastTransform.jl/dev/)
[![Build Status](https://github.com/MagineZ/QuestionnaireFastTransform.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MagineZ/QuestionnaireFastTransform.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MagineZ/QuestionnaireFastTransform.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MagineZ/QuestionnaireFastTransform.jl)

This code is built in Julia environment and calls Python 'pyquest' package.

We build more features based on the original 'pyqyest': https://github.com/gmishne/pyquest

This version of pyquest includes new features:
- Cosine/Correlation Affinity
- multi-scale Cosine/Correlation Affinity
- Landmark diffusion map 

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
```
## Reference
- P.-C. Su and R. R. Coifman, "Learning the Analytic Geometry of Transformations to Achieve Efficient Computation," arXiv preprint arXiv:2506.11990, 2025.

- N. Saito and Y. Shao, "eGHWT: The Extended Generalized Haar–Walsh Transform," *Journal of Mathematical Imaging and Vision*, vol. 64, no. 3, pp. 261–283, 2022.

- G. Mishne, R. Talmon, I. Cohen, R. R. Coifman and Y. Kluger, "Data-Driven Tree Transforms and Metrics," IEEE Transactions on Signal and Information Processing over Networks, vol. 4, no. 3, pp. 451–466, 2017.

- G. Mishne, R. Talmon, R. Meir, J. Schiller, U. Dubin and R. R. Coifman, "Hierarchical Coupled Geometry Analysis for Neuronal Structure and Activity Pattern Discovery," IEEE Journal of Selected Topics in Signal Processing, vol. 10, no. 7, pp. 1238-1253, Oct. 2016.

-J. I. Ankenman, “Geometry and analysis of dual networks on questionnaires,” Ph.D. dissertation, Yale University, 2014.

- M. O'Neil, F. Woolfe and V. Rokhlin, "An Algorithm for the Rapid Evaluation of Special Function Transforms," *Applied and Computational Harmonic Analysis*, vol. 28, no. 2, pp. 203–226, 2010.

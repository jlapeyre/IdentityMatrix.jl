# IdentityMatrix

[![Build Status](https://github.com/jlapeyre/IdentityMatrix.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jlapeyre/IdentityMatrix.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/jlapeyre/IdentityMatrix.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/jlapeyre/IdentityMatrix.jl)

This package provides `Id{T, N}`, representing the `N`x`N` identity matrix of element type `T`.

This implementation is less complicated and somewhat more performant for some tasks than `LinearAlgebra.UniformScaling` and
`FillArrays.Eye`. In particular, multiplication of small matrices (say 2x2) is significantly faster.

This package is meant to be used when you do a lot of work with matrices of a few fixed sizes. The size of the identity matrix
is encoded in the type. This means that methods will be recompiled for each size.

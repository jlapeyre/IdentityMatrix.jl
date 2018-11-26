# IdentityMatrix

[![Build Status](https://travis-ci.com/jlapeyre/IdentityMatrix.jl.svg?branch=master)](https://travis-ci.com/jlapeyre/IdentityMatrix.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/jlapeyre/IdentityMatrix.jl?svg=true)](https://ci.appveyor.com/project/jlapeyre/IdentityMatrix-jl)
[![Codecov](https://codecov.io/gh/jlapeyre/IdentityMatrix.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jlapeyre/IdentityMatrix.jl)
[![Coveralls](https://coveralls.io/repos/github/jlapeyre/IdentityMatrix.jl/badge.svg?branch=master)](https://coveralls.io/github/jlapeyre/IdentityMatrix.jl?branch=master)

This package implements several methods specialized for `FillArrays.Eye`.
Some of the new methods compute the output at compile time.

Without these new methods, the type `FillArrays.Eye` relies on methods for `Diagonal`,
many of which are very inefficient for an identity matrix.

To use, load the module
```julia
using FillArrays
using IdentityMatrix
```

Also provided are:

* `identitymatrix(T, n)`, which is faster than `Matrix{T}(I, n, n)`

*  More efficent methods for `kron(::AbstractArray, ::Diagonal)` and `kron(::Diagonal, ::AbstractArray)`. Note
   that this applies to any `Diagonal{T<:Number}`, not just identity matrices.

Here is an incomplete list of methods that are improved over the fallbacks for `Eye`.

* `iterate`

* `copy`, `Matrix`, (maybe)

* `kron(a, b)`, where either or both of `a` and `b` is an identity matrix

* `eigen`, `eigvecs`, `eigvals`

* `IM::Idents / A::AbstractMatrix`

* `A::AbstractMatrix / IM::Idents`

* `IM::Idents \ A::AbstractMatrix`

* `A::AbstractMatrix \ IM::Idents`

* `IM::Idents * A::AbstractMatrix`

* `A::AbstractMatrix * IM::Idents`

* `diag`

* `isposdef`

* `first`, `last`, `minimum`, `maximum`, `extrema`

* `permutedims`, `triu` `triu!`, `tril`,  `tril`, `inv`

* `det`, `logdet`

* `imag`

* `^(IM::Idents, p::Integer)`

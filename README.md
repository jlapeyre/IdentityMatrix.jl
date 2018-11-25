# IdentityMatrix

[![Build Status](https://travis-ci.com/jlapeyre/IdentityMatrix.jl.svg?branch=master)](https://travis-ci.com/jlapeyre/IdentityMatrix.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/jlapeyre/IdentityMatrix.jl?svg=true)](https://ci.appveyor.com/project/jlapeyre/IdentityMatrix-jl)
[![Codecov](https://codecov.io/gh/jlapeyre/IdentityMatrix.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jlapeyre/IdentityMatrix.jl)
[![Coveralls](https://coveralls.io/repos/github/jlapeyre/IdentityMatrix.jl/badge.svg?branch=master)](https://coveralls.io/github/jlapeyre/IdentityMatrix.jl?branch=master)

This package implements a matrix-identity type and several methods specialized for this type and for `FillArrays.Eye`.

`Identity(n)` returns an `n` x `n` identity matrix. 

Compared to the corresponding "dense" matrix, many operations are more efficient in both time and storage.
Often, the output can be computed at compile time.
The type `FillArrays.Eye` relies on methods for `Diagonal`,
many of which are very inefficient for an identity matrix.
The methods in this package take type `Idents{T} = Union{Eye{T}, Identity{T}}`
if doing so improves performance.

In the future, the organization of the identity matrix types and corresponding methods
will certainly differ from this package. For instance:
1) The output object types may be changed for some methods.
2) All the methods may be moved to `FillArrays`,
and `Eye` will be the only identity matrix type.
3) The methods may take traits rather than a Union type.

Here is an incomplete list of methods that are improved over the fallbacks for `Eye`.

* `iterate`

* `copy`, `Matrix`

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

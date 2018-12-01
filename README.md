# IdentityMatrix

[![Build Status](https://travis-ci.com/jlapeyre/IdentityMatrix.jl.svg?branch=master)](https://travis-ci.com/jlapeyre/IdentityMatrix.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/jlapeyre/IdentityMatrix.jl?svg=true)](https://ci.appveyor.com/project/jlapeyre/IdentityMatrix-jl)
[![Codecov](https://codecov.io/gh/jlapeyre/IdentityMatrix.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jlapeyre/IdentityMatrix.jl)
[![Coveralls](https://coveralls.io/repos/github/jlapeyre/IdentityMatrix.jl/badge.svg?branch=master)](https://coveralls.io/github/jlapeyre/IdentityMatrix.jl?branch=master)

NOTE: This package is being canibalized; Pieces moved to `FillArrays.jl` and `LinearAlgebra`. If you don't have all these
in sync, you make get warnings or errors about overridden methods.
But, the master branches of `IdentityMatrix.jl` and `FillArrays.jl` will be kept in sync.

This package implements several methods specialized for types `Diagonal`,
`FillArrays.Fill`, and `FillArrays.Eye`. They are more efficient, often much more,
than the fallback methods.

The methods are more-or-less drop-in replacements.

To use, load the module
```julia
using FillArrays
using IdentityMatrix
```

Also provided are:

* `idmat(T, n)`, which is (sometimes) a bit faster than `Matrix{T}(I, n, n)`. If you want efficiency
you should benchmark them for your use case, or look at the code.

Here is an incomplete list of methods that are improved over the fallbacks for `Eye`.

* `iterate`

* `copy`, `Matrix`

* `kron(a, b)`, where either or both of `a` and `b` is an identity matrix or a `Diagonal` matrix.

* `IM::Idents / A::AbstractMatrix`

* `A::AbstractMatrix / IM::Idents`

* `IM::Idents \ A::AbstractMatrix`

* `A::AbstractMatrix \ IM::Idents`

* `IM::Idents * A::AbstractMatrix`

* `A::AbstractMatrix * IM::Idents`

*  Matrix operations with `UniformScaling`

* `^(IM::Idents, p::Integer)`

* `eigen`, `eigvecs`, `eigvals`

* `isposdef`, `imag`

* `diag`, `first`, `last`, `minimum`, `maximum`, `extrema`, `any`, `all`, `sum`, `prod`

* `norm`, `opnorm`

* `permutedims`, `triu` `triu!`, `tril`,  `tril`, `inv`

* `det`, `logdet`

<!--  LocalWords:  IdentityMatrix Codecov FillArrays julia idmat fallbacks kron
 -->
<!--  LocalWords:  UniformScaling eigen eigvecs eigvals isposdef imag diag triu
 -->
<!--  LocalWords:  extrema opnorm permutedims tril inv det logdet
 -->

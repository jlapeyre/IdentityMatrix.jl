"""
    module IdentityMatrix

This module provides the type `IdentityMatrix`.
An identity matrix with element type `Float64` is constructed with any of
```
Identity{Float64}(n)
Identity(n)
eye(Float64, n)
eye(n)
```

Objects of this type are meant to act for the most part like the corresponding dense matrix.
Use `Matrix(eye(n))` to obtain a dense identity matrix.
"""
module IdentityMatrix
import LinearAlgebra
using FillArrays

export Identity, eye, isorthogonal, isunitary, Idents

struct Identity{T<:Any} <: AbstractMatrix{T}
    ncols::Int
end

Idents{T} = Union{Identity{T}, Eye{T}}

function Base.size(IM::Identity, d::Integer)
    d < 1 && throw(ArgumentError("dimension must be ≥ 1, got $d"))
    return d <= 2 ? IM.ncols : 1
end
Base.size(IM::Identity) = (IM.ncols, IM.ncols)

checkuniquedim(IM::Idents) = size(IM, 1)
checkuniquedim(D::LinearAlgebra.Diagonal) = size(D, 1)
checkuniquedim(AV::AbstractVector) = length(AV)

function checkuniquedim(A::AbstractMatrix)
    m, n = size(A)
    m == n || throw(DimensionMismatch("matrix is not square: dimensions are $(size(A))"))
    return m
end

function checkuniquedim(A, B)
    m1 = checkuniquedim(A)
    m2 = checkuniquedim(B)
    m1 == m2 || throw(DimensionMismatch("matrix A has dimensions $(size(A)). matirix B has dimensions $(size(B))"))
    return m1
end

## constructors

Identity(n::Integer) = Identity{Float64}(n)
eye(::Type{T}, n::Integer) where T = Identity{T}(n)
eye(n::Integer) = eye(Float64, n)
function eye(::Type{T}, AM::AbstractMatrix) where T
    dim = checkuniquedim(AM)
    return Identity{T}(dim)
end
eye(AM::AbstractMatrix) = eye(eltype(AM), AM)

## conversion

Identity(IM::Identity) = IM
Base.AbstractMatrix{T}(IM::Identity) where {T} = Identity{T}(size(IM, 1))

# This is recommended in the manual. But the function `materialize` below
# is significantly faster, especially for small matrices.
#Base.Matrix(IM::Identity) = Matrix{eltype(IM)}(LinearAlgebra.I, size(IM)...)

Base.Matrix(IM::Identity) = materialize(IM)
Base.Array(IM::Identity) = Matrix(IM)

Identity(D::LinearAlgebra.Diagonal) = all(isone, D.diag) ? Identity{eltype(D)}(size(D,1)) :
    throw(InexactError(:Identity, Identity{eltype(D)}, D))

import LinearAlgebra.diag
diag(IM::Idents{T}) where T = ones(T, size(IM, 1))
LinearAlgebra.Diagonal(IM::Identity{T}) where T = LinearAlgebra.Diagonal(diag(IM))

Base.eltype(::Type{Identity{T}}) where T = T
Base.eltypeof(::Identity{T}) where T = T

@inline function Base.getindex(IM::Identity{T}, i::Integer, j::Integer) where T
    @boundscheck Base.checkbounds(IM, i, j)
    return i == j ? one(eltype(IM)) : zero(eltype(IM))
end

function Base.replace_in_print_matrix(A::Identity, i::Integer, j::Integer, s::AbstractString)
    i == j ? s : Base.replace_with_centered_mark(s)
end

# The fallback method for AbstractArray is already efficient for Idents for:
# length, firstindex, lastindex, axes, checkbounds

# The fallback method (for Diagonal) is already efficient for Eye for:
# getindex, factorize, real, float, ishermitian, issymmetric,
# isdiag, istriu, istril, log

LinearAlgebra.factorize(IM::Identity) = IM

Base.real(IM::Identity{T}) where T = IM
# The default method for Diagonal returns Diagonal with a dense diagonal
Base.imag(IM::Idents{T}) where T = LinearAlgebra.Diagonal(Fill(zero(T), size(IM, 1)))
Base.float(IM::Identity{T}) where T = Identity{float(T)}(size(IM, 1))

LinearAlgebra.ishermitian(::Identity) = true
LinearAlgebra.issymmetric(::Identity) = true
LinearAlgebra.isposdef(::Idents) = true

# These may not be useful, since they don't exist in LinearAlgebra.
isorthogonal(::Idents) = true
isunitary(::Idents) = true

Base.iszero(::Idents) = false
Base.isone(::Idents) = true
LinearAlgebra.isdiag(::Identity) = true
LinearAlgebra.istriu(::Identity) = true
LinearAlgebra.istril(::Identity) = true

Base.sum(IM::Idents) = convert(eltype(IM), size(IM, 1))
Base.prod(IM::Idents) = zero(eltype(IM))

# This is faster than Matrix(I, n, n)
function materialize(IM::Idents)
    a = zeros(eltype(IM), size(IM))
    @inbounds for i in 1:size(IM, 1) # I think the compiler uses @inbounds here regardless
        a[i, i] = 1  # A bit faster than linear indexing
    end
    return a
end

# For Eye{T}, only the diagonal is materialized. Here we create a dense matrix.
Base.copymutable(IM::Identity) = materialize(IM)

Base.first(::Idents{T}) where T = one(T)
Base.last(::Idents{T}) where T = one(T)
Base.minimum(::Idents{T}) where T = zero(T)
Base.maximum(::Idents{T}) where T = one(T)
Base.extrema(IM::Idents) = (minimum(IM), maximum(IM)) # FIXME: implement extrema(IM, dims = dims)

import Base: transpose, adjoint, conj, inv, permutedims
import LinearAlgebra: triu, triu!, tril, tril!
for f in (:transpose, :adjoint, :conj) # These are already effcient for Eye
    @eval ($f)(IM::Identity) = IM
end

for f in (:permutedims, :triu, :triu!, :tril, :tril!, :inv) # These override inefficient methods for Eye
    @eval ($f)(IM::Idents) = IM
end

# This agrees with diag for dense matrices in that any integer `k` is allowed.
# It disagrees with diag for Diagonal, which throws and error when `abs(k)` is too large.
function LinearAlgebra.diag(IM::Idents{T}, k::Integer) where T
    k == 0 && return LinearAlgebra.diag(IM)
    m = size(IM, 1)
    (k > m || k < -m) && return T[]
    return zeros(T, m - abs(k))
end

LinearAlgebra.tr(IM::Identity) = sum(IM)
LinearAlgebra.det(::Idents{T}) where T = one(T)
LinearAlgebra.logdet(::Idents{T}) where T = log(one(T))
Base.sqrt(IM::Idents) = IM
Base.log(IM::Identity) = LinearAlgebra.Diagonal(Fill(zero(eltype(IM)), size(IM,1)))

# Matrix functions
import Base: exp, cos, sin, tan, csc, sec, cot, cosh, sinh, tanh, csch, sech,
    coth, acos, asin, atan, acsc, asec, acot, acosh, asinh, atanh, acsch, asech, acoth
for f in (:exp, :cos, :sin, :tan, :csc, :sec, :cot,
          :cosh, :sinh, :tanh, :csch, :sech, :coth,
          :acos, :asin, :atan, :acsc, :asec, :acot,
          :acosh, :asinh, :atanh, :acsch, :asech, :acoth)
    @eval $f(IM::Identity) = LinearAlgebra.Diagonal(Fill($f(one(eltype(IM))), size(IM, 1)))
end

# There are a few other functions `f` in the list above for which `f(IM)` cannot be computed
# at compile time. This is one way to solve this problem.
const _cschval = csch(1.0)
mycsch(::Union{Type{Float64}, Type{Int}}) = _cschval
mycsch(::Type{T}) where T = csch(one(T))
csch(IM::Identity) = LinearAlgebra.Diagonal(Fill(mycsch(eltype(IM)), size(IM, 1)))

Base.:(^)(IM::Idents, p::Integer) = IM
Base.:(==)(IMa::Identity, IMb::Identity) = size(IMa, 1) == size(IMb, 1)
(Base.:-)(IM::Identity{T}) where T = LinearAlgebra.Diagonal(-diag(IM))

function (Base.:+)(IMa::Identity{T}, IMb::Identity{V}) where {T,V}
    d = checkuniquedim(IMa, IMb)
    LinearAlgebra.Diagonal(Fill(one(T) + one(V), d))
end

function (Base.:-)(IMa::Identity{T}, IMb::Identity{V}) where {T,V}
    d = checkuniquedim(IMa, IMb)
    zeros(Base.promote_op(-, T, V), size(IMa))
end

# Diagonal is already efficient
(Base.:*)(x::Number, IM::Identity{V}) where V = FillArrays.Fill{Base.promote_op(*, typeof(x), V)}(x, size(IM, 1)) |> LinearAlgebra.Diagonal
(Base.:*)(IM::Identity, x::Number) = x * IM  # Diagonal is already efficient
function (Base.:/)(IM::Identity{T}, x::Number) where T # Diagonal is already efficient
    return LinearAlgebra.Diagonal(Fill(one(T) / x, size(IM, 1)))
end

function (Base.:*)(IMa::Identity{T}, IMb::Identity{V}) where {T, V}
    d = checkuniquedim(IMa, IMb)
    return Identity{Base.promote_op(*, T, V)}(d)
end

# Diagonal is already efficient
(Base.:/)(IMa::Identity{T}, IMb::Identity{V}) where {T, V} = IMa * IMb

# Diagonal is already efficient
(Base.:/)(AM::AbstractMatrix, IM::Identity) = IM * AM

function (Base.:*)(IM::Identity{T}, AV::AbstractVector{T}) where T
    junk = checkuniquedim(IM, AV)
    return AV
end

function (Base.:*)(IM::Identity{T}, AV::AbstractVector{V}) where {T, V}
    junk = checkuniquedim(IM, AV)
    return one(T) .* AV
end

function (Base.:*)(IM::Identity{T}, AM::AbstractMatrix{V}) where {T,V}
    junk = checkuniquedim(IM, AM)
    return convert(Matrix{Base.promote_op(*, T, V)}, AM)
end

(Base.:*)(IM::Identity{T}, AM::AbstractMatrix{T}) where T = (checkuniquedim(IM, AM); AM)
(Base.:*)(AM::AbstractMatrix{T}, IM::Identity{T}) where T = IM * AM

function Base.iterate(iter::Idents{T}, istate = (1, 1)) where T
    (i::Int, j::Int) = istate
    m = size(iter, 1)
    return i > m ? nothing :
        ((@inbounds getindex(iter, i, j)),
         j == m ? (i + 1, 1) : (i, j + 1))
end

# Object on left should be factorization object
# LinearAlgebra.ldiv!(IM::Identity{T}, AV::AbstractVector{T}) where {T} = one(T) .* AV
# LinearAlgebra.ldiv!(IM::Identity{T}, AM::AbstractMatrix{T}) where {T} = one(T) .* AM

function Base.kron(a::Idents{T}, b::AbstractMatrix{S}) where {T, S}
    @assert ! Base.has_offset_axes(b)
    R = zeros(Base.promote_op(*, T, S), size(a, 1) * size(b, 1), size(a, 2) * size(b, 2))
    m = 1
    for j = 1:size(a, 2)
        for l = 1:size(b, 2)
            for k = 1:size(b, 1)
                R[m] = b[k,l]
                m += 1
            end
            m += (size(a, 2) - 1) * size(b, 2)
        end
        m += size(b, 2)
    end
    return R
end

function Base.kron(a::AbstractMatrix{T}, b::Idents{S}) where {T, S}
    @assert ! Base.has_offset_axes(a)
    R = zeros(Base.promote_op(*, T, S), size(a, 1) * size(b, 1), size(a, 2) * size(b, 2))
    m = 1
    for j = 1:size(a, 1)
        for l = 1:size(b, 1)
            for k = 1:size(a, 2)
                R[m] = a[k,j]
                m += size(b, 2)
            end
            m += 1
        end
        m -= size(b, 2)
    end
    return R
end

"""
    idents(from, T,  n)

Create an identity matrix object of type `typeof(from)`, with new
data type `T` and ncols `n`.
"""
idents(from::Identity, T,  n) = Identity{T}(n)
idents(from::Eye, T, n) = Eye{T}(n)

Base.kron(a::Idents{T}, b::Idents{T}) where T = idents(a, T, size(a, 1) * size(b, 1))
Base.kron(a::Idents{T}, b::Idents{V}) where {T, V} = idents(a, Base.promote_op(*, T, V), (size(a, 1) * size(b, 1)))

LinearAlgebra.eigvals(IM::Idents{T}) where T = diag(IM)
LinearAlgebra.eigvecs(IM::Idents) = IM
LinearAlgebra.eigen(IM::Idents) = LinearAlgebra.Eigen(LinearAlgebra.eigvals(IM), LinearAlgebra.eigvecs(IM))

ensuretype(::Type{T}, AM::AbstractArray{T}) where {T} =  AM
ensuretype(::Type{T}, AM::AbstractArray{V}) where {T, V} = one(T) .* AM

# Put these last. The backslash confuses emacs.
# Diagonal is already efficient. But, we use `Idents` to remove fatal method ambiguity intrduced
# by the methods below.
(Base.:\)(IMa::Idents{T}, IMb::Idents{V}) where {T, V} = IMa * IMb

(Base.:\)(AM::AbstractMatrix{T}, IM::Idents{V}) where {T, V} = ensuretype(Base.promote_op(*, T, V), inv(AM))
(Base.:\)(IM::Idents{V}, AM::AbstractMatrix{T}) where {T, V} = ensuretype(Base.promote_op(*, T, V), AM)

end # module

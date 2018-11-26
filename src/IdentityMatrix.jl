"""
    module IdentityMatrix

This module overrides more generic methods for identity matrices.
The methods are immediately available after loading.
The methods apply to both `FillArrays.Eye`
and to `IdentityMatrix.Identity`.
"""
module IdentityMatrix

import LinearAlgebra
using FillArrays

export Identity, eye, isorthogonal, isunitary, Idents

"""
    struct Identity{T<:Any} <: AbstractMatrix{T}

Provides an alternative to `FillArrays.Eye` for a lazy identity matrix.
"""
struct Identity{T<:Any} <: AbstractMatrix{T}
    ncols::Int
end

"""
    Idents{T} = Union{Identity{T}, Eye{T}}

Identity matrix types.
Methods in `IdentityMatrix.jl` may break some usage of `Eye` elsewhere.
By remvoving `Eye` from `Idents{T}`,
you can isolate these methods from the rest of the system.
The performance of the methods defined for `Idents` is identical for `Eye` and `Identity`.

Methods not defined for `Idents` fall back to `AbstractMatrix` for `Identity` and to
`Diagonal` for `Eye`.
"""
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

## constructors for Identity

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

LinearAlgebra.Diagonal(IM::Identity{T}) where T = LinearAlgebra.Diagonal(diag(IM))

Base.eltype(::Type{Identity{T}}) where T = T
Base.eltypeof(::Identity{T}) where T = T

# Already efficient for Eye
@inline function Base.getindex(IM::Identity{T}, i::Integer, j::Integer) where T
    @boundscheck Base.checkbounds(IM, i, j)
    return i == j ? one(eltype(IM)) : zero(eltype(IM))
end

function Base.replace_in_print_matrix(A::Identity, i::Integer, j::Integer, s::AbstractString)
    i == j ? s : Base.replace_with_centered_mark(s)
end

include("Eye.jl")

LinearAlgebra.factorize(IM::Identity) = IM

Base.real(IM::Identity{T}) where T = IM
Base.float(IM::Identity{T}) where T = Identity{float(T)}(size(IM, 1))

LinearAlgebra.ishermitian(::Identity) = true
LinearAlgebra.issymmetric(::Identity) = true

# These may not be useful, since they don't exist in LinearAlgebra.
isorthogonal(::Idents) = true
isunitary(::Idents) = true

LinearAlgebra.isdiag(::Identity) = true
LinearAlgebra.istriu(::Identity) = true
LinearAlgebra.istril(::Identity) = true

import Base: transpose, adjoint, conj
import Base: inv, permutedims
import LinearAlgebra: triu, triu!, tril, tril!
for f in (:transpose, :adjoint, :conj) # These are already effcient for Eye
    @eval ($f)(IM::Identity) = IM
end

LinearAlgebra.tr(IM::Identity) = sum(IM)
Base.log(IM::Identity) = LinearAlgebra.Diagonal(Fill(zero(eltype(IM)), size(IM,1)))

# Matrix functions
import Base: exp, cos, sin, tan, csc, sec, cot, cosh, sinh, tanh, csch, sech,
    coth, acos, asin, atan, acsc, asec, acot, acosh, asinh, atanh, acsch, asech, acoth
for f in (:exp, :cos, :sin, :tan, :csc, :sec, :cot,
          :cosh, :sinh, :tanh, :sech, :coth,
          :acos, :asin, :atan, :acsc, :asec, :acot,
          :acosh, :asinh, :atanh, :acsch, :asech, :acoth)
    @eval $f(IM::Identity) = LinearAlgebra.Diagonal(Fill($f(one(eltype(IM))), size(IM, 1)))
end

Base.:(==)(IMa::Identity, IMb::Identity) = size(IMa, 1) == size(IMb, 1)
(Base.:-)(IM::Identity{T}) where T = -FillArrays.Eye{T}(size(IM, 1))

# Diagonal is already efficient
# But, we must disambiguate
function (Base.:+)(IMa::Idents{T}, IMb::Identity{V}) where {T,V}
    d = checkuniquedim(IMa, IMb)
    return LinearAlgebra.Diagonal(Fill(one(T) + one(V), d))
end

# This is a bit slow. But, we will discard it anyway.
# Diagonal is already efficient
function (Base.:-)(IMa::Identity{T}, IMb::Identity{V}) where {T,V}
    d = checkuniquedim(IMa, IMb)
    return Diagonal(Zeros{Base.promote_op(-, T, V)}(size(IMa)))
end

# Diagonal is already efficient
(Base.:*)(x::Number, IM::Identity{V}) where V = FillArrays.Fill{Base.promote_op(*, typeof(x), V)}(x, size(IM, 1)) |> LinearAlgebra.Diagonal

# Diagonal is already efficient
(Base.:*)(IM::Identity, x::Number) = x * IM

# Diagonal is already efficient
function (Base.:/)(IM::Identity{T}, x::Number) where T
    return LinearAlgebra.Diagonal(Fill(one(T) / x, size(IM, 1)))
end

# Diagonal is already efficient
function (Base.:*)(IMa::Identity{T}, IMb::Identity{V}) where {T, V}
    d = checkuniquedim(IMa, IMb)
    return Identity{Base.promote_op(*, T, V)}(d)
end

# Diagonal is already efficient Not needed
#(Base.:/)(IMa::Identity{T}, IMb::Identity{V}) where {T, V} = IMa * IMb

# Object on left should be factorization object
# LinearAlgebra.ldiv!(IM::Identity{T}, AV::AbstractVector{T}) where {T} = one(T) .* AV
# LinearAlgebra.ldiv!(IM::Identity{T}, AM::AbstractMatrix{T}) where {T} = one(T) .* AM



end # module

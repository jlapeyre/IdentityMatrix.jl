module IdentityMatrix

# UniformScaling is much less performant than it could be when the
# scale factor is not 1.
#
# TODO move this to a separate package

import LinearAlgebra
using LinearAlgebra: Diagonal
export Id

"""
    Id{T, N}

The `N`x`N` identity matrix with element type `T`.
"""
struct Id{T, N} <: AbstractMatrix{T}
end

Id(n::Integer) = Id{Float64, n}()
Id(::Type{T},  n::Integer) where T = Id{T, n}()

# Id is optimally indexed with cartesian indices
Base.IndexStyle(::Type{<:Id}) = Base.IndexCartesian()
Base.size(::Id{T, N}) where {T, N} = (N, N)
# These are singletons. We could perhaps make this a method error
Base.copy(m::Id)= m

Base.complex(m::Id{T, N}) where {T, N} = Id(complex(T), N)
Base.float(m::Id{T, N}) where {T, N} = Id(float(T), N)

# Fallback is efficient
# Base.length

# We could use the fallback method for `one`.
# But here N is known at compile time.
function Base.collect(m::Id{T, N}) where {T, N}
    mout = zeros(T, (N, N))
    for i in 1:N
        mout[i, i] = one(T)
    end
    mout
end

function Base.getindex(::Id{T, N}, i::Integer, j::Integer) where {T, N}
    i == j && return one(T)
    return zero(T)
end

# function Base.getindex(::Id{T, N}, i::Integer) where {T, N}
#     zero(T)
# end

function Base.:*(::Id{T, N}, m::AbstractMatrix{T}) where {T, N}
    mB = size(m, 1)
    mB == N || throw(DimensionMismatch(lazy"A has dimensions ($N,$N) but B has first dimension $mB"))
    copy(m)
end

function Base.:*(::Id{T, N}, m::AbstractMatrix{T2}) where {T, N, T2}
    mB = size(m, 1)
    mB == N || throw(DimensionMismatch(lazy"A has dimensions ($N,$N) but B has first dimension $mB"))
    Matrix{promote_type(T, T2)}(m)
end

Base.:*(::Id{T, N}, ::Id{T, N}) where {T, N} = Id{T, N}()

function Base.:*(::Id{T, N}, m::Diagonal{T}) where {T, N}
    mB = size(m, 1)
    mB == N || throw(DimensionMismatch(lazy"A has dimensions ($N,$N) but B has first dimension $mB"))
    copy(m)
end

function Base.:*(::Id{T, N}, m::Diagonal{T2}) where {T, N, T2}
    mB = size(m, 1)
    mB == N || throw(DimensionMismatch(lazy"A has dimensions ($N,$N) but B has first dimension $mB"))
    Tp = promote_type(T, T2)
    Diagonal(convert(Vector{Tp}, m.diag))
end

Base.:*(md::Diagonal, mid::Id) = mid * md

# Separate method from the general case because `collect` is faster
# than "converting" to same type.
function Base.:+(::Id{T, N}, m::AbstractMatrix{T}) where {T, N}
    mout = collect(m)
    for i in 1:N
        mout[i, i] += one(T)
    end
    mout
end

function Base.:+(::Id{T, N}, m::AbstractMatrix{T2}) where {T, N, T2}
    mout = Matrix{promote_type(T, T2)}(m)
    for i in 1:N
        mout[i, i] += one(T)
    end
    mout
end


Base.:+(m::AbstractMatrix, mid::Id) = mid + m

Base.:*(m::AbstractMatrix, mi::Id) = mi * m
Base.sum(::Id{T,N}) where {T, N} = N * one(T)
Base.prod(::Id{T}) where T = zero(T)
Base.prod(::Id{T, 1}) where T = one(T)
Base.inv(m::Id) = m
Base.one(m::Id) = m

# Fallback is efficient
# Base.zero
# Base.conj

# We could depend on FillArrays, and return here type One
LinearAlgebra.diag(::Id{T, N}) where {T, N} = ones(T, N)
LinearAlgebra.eigvals(m::Id) = LinearAlgebra.diag(m)
LinearAlgebra.eigen(m::Id) = LinearAlgebra.Eigen(LinearAlgebra.eigvals(m), m)
Base.transpose(m::Id) = m
Base.adjoint(m::Id) = m

Base.:^(m::Id, p::Integer) = m

end

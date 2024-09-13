module IdentityMatrix

# TODO:
#
# kron! -- most important
# kron! is partially done.
# Base.:\
# eachrow, eachcol
# map

# For compiling workflows for statically-compiled-like latency
using PrecompileTools: @setup_workload, @compile_workload

import LinearAlgebra
using LinearAlgebra: Diagonal
export Id

"""
    Id{T, N}

The `N`x`N` identity matrix with element type `T`.
"""
struct Id{T, N} <: AbstractMatrix{T}
end

"""
    Id(n::Integer)

Return Id{Float64, n}().
"""
Id(n::Integer) = Id{Float64, n}()

"""
    Id(Complex, n::Integer)

Return Id{ComplexF64, n}().
"""
Id(::Type{Complex}, n::Integer) = Id(ComplexF64, n)


"""
    Id(::Type{T}, n::Integer)

Return Id{T, n}().
"""
Id(::Type{T},  n::Integer) where T = Id{T, n}()

#Base.show(x::Id) = show(stdout, x)

function Base.show(io::IO, x::Id)
    print(io, "Id(", size(x,1))
    if size(x,1) != size(x,2)
        print(io, ",", size(x,2))
    end
    print(io, ")")
end

Base.array_summary(io::IO, a::Id{T, N}, inds::Tuple{Vararg{Base.OneTo}}) where {T, N} =
    print(io, Base.dims2string(length.(inds)), " Id{$T, $N}")

# There is no point to representing the elements of the identity matrix.
# So we show minimal information.
function Base.show(io::IO, ::MIME"text/plain", x::Id)
    if get(IOContext(io), :compact, false)  # for example [Fill(i==j,2,2) for i in 1:3, j in 1:4]
        return show(io, x)
    end
    summary(io, x)
end

# Id is optimally indexed with cartesian indices
Base.IndexStyle(::Type{<:Id}) = Base.IndexCartesian()
Base.size(::Id{T, N}) where {T, N} = (N, N)
# These are singletons. We could perhaps make this a method error
Base.copy(m::Id) = m

for func in (:complex, :float, :big)
    @eval Base.$func(m::Id{T, N}) where {T, N} = Id(($func)(T), N)
end

Base.real(::Id{T, N}) where {T, N} = Id(real(T), N)

# Fallback is efficient
# Base.real(::Id{<:Real})
# Base.length
# Base.eltype

# We could use the fallback method for `one`.
# But here N is known at compile time.
function Base.collect(m::Id{T, N}) where {T, N}
    mout = zeros(T, (N, N))
    for i in 1:N
        mout[i, i] = one(T)
    end
    mout
end

@inline function Base.getindex(mi::Id{T, N}, i::Integer, j::Integer) where {T, N}
    @boundscheck checkbounds(mi, i, j)
    i == j && return one(T)
    return zero(T)
end

@inline function Base.getindex(m::Id{T, N}, ilin::Integer) where {T, N}
    (i, j) = divrem(ilin - 1, N)
    m[i+1, j+1]
end

function Base.:*(::Id{T, N}, m::AbstractMatrix{T}) where {T, N}
    mB = size(m, 1)
    mB == N || throw(DimensionMismatch(lazy"A has dimensions ($N,$N) but B has first dimension $mB"))
    copy(m)
end

function Base.:*(::Id{T, N}, m::AbstractMatrix{T2}) where {T, N, T2}
    mB = size(m, 1)
    mB == N || throw(DimensionMismatch(lazy"A has dimensions ($N,$N) but B has first dimension $mB"))
    convert(AbstractArray{promote_type(T, T2)}, m) # This is evidently faster
#    Matrix{promote_type(T, T2)}(m)
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

function Base.:/(mid::Id{T,N}, m::AbstractMatrix{T}) where {T, N}
    mB = size(m, 2)
    mB == N || throw(DimensionMismatch(lazy"A has dimensions ($N,$N) but B has second dimension $mB"))
    LinearAlgebra.pinv(m)
end

Base.:*(md::Diagonal, mid::Id) = mid * md
Base.:*(c::Number, mid::Id{T, N}) where {T, N} = LinearAlgebra.UniformScaling(c * one(T))(N)
Base.:*(mid::Id{T, N}, c::Number) where {T, N} = c * mid
Base.:/(mid::Id{T, N}, c::Number) where {T, N} = LinearAlgebra.UniformScaling(one(T) / c)(N)

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
    mout = convert(AbstractArray{promote_type(T, T2)}, m) # This is evidently faster
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

for f in (:permutedims, :triu, :triu!, :tril, :tril!, :inv, :one, :eigvecs,
          :transpose, :adjoint, :sqrt)
    @eval LinearAlgebra.$f(M::Id) = M
end

# We could depend on FillArrays, and return here type One
#LinearAlgebra.diag(::Id{T, N}) where {T, N} = ones(T, N)
# This agrees with diag for dense matrices in that any integer `k` is allowed.
# It disagrees with diag for Diagonal, which throws an error when `abs(k)` is too large.
function LinearAlgebra.diag(IM::Id{T}, k::Integer) where T
    k == 0 && return LinearAlgebra.diag(IM)
    m = size(IM, 1)
    (k > m || k < -m) && return T[]
    return zeros(T, m - abs(k))
end
LinearAlgebra.diag(m::Id{T, N}) where {T, N} = ones(T, N)

LinearAlgebra.eigvals(m::Id) = LinearAlgebra.diag(m)
LinearAlgebra.eigen(m::Id) = LinearAlgebra.Eigen(LinearAlgebra.eigvals(m), m)
LinearAlgebra.eigmin(::Id{T}) where T = one(T)
LinearAlgebra.eigmax(::Id{T}) where T = one(T)
Base.:^(m::Id, p::Integer) = m

# The default implementations of isone and iszero are completely naive.
# But they return the correct result instantly no matter the size of `Id`.
# Still, we define these.
Base.isone(::Id{<:Number}) = true
Base.iszero(::Id{<:Number}) = false

Base.in(x, ::Id{T, 1}) where T = x == one(T)
Base.in(x, ::Id{T, N}) where {T, N} = x == one(T) || x == zero(T)
Base.issubset(v, ::Id{T, N}) where {T, N} = all(x -> x == one(T) || x == zero(T), v)
Base.issubset(v, ::Id{T, 1}) where T = all(x -> x == one(T), v)
Base.isreal(::Id) = true
LinearAlgebra.isposdef(::Id) = true

# Fallback is efficient
# Base.zero
# Base.conj
# Base.any(::Id)
# Base.all(::Id)

# Performance is the same as for FillArrays.Eye
@inline function Base.kron!(C::AbstractMatrix, A::Id, B::AbstractMatrix)
    Base.require_one_based_indexing(B)
    (mA, nA) = size(A)
    (mB, nB) = size(B)
    (mC, nC) = size(C)
    @boundscheck (mC, nC) == (mA * mB, nA * nB) ||
        throw(DimensionMismatch("expect C to be a $(mA * mB)x$(nA * nB) matrix, got size $(mC)x$(nC)"))
    isempty(A) || isempty(B) || fill!(C, zero(A[1,1] * B[1,1]))
    m = 1
    @inbounds for j = 1:nA
        for k = 1:nB
            for l = 1:mB
                C[m] = B[l,k]
                m += 1
            end
            m += (nA - 1) * mB
        end
        m += mB
    end
    return C
end

Base.kron(a::Id{T}, b::Id{T}) where {T<:Number} = Id(T, size(a, 1) * size(b, 1))
Base.kron(a::Id{T}, b::Id{V}) where {T<:Number, V<:Number} = Id(Base.promote_op(*, T, V), size(a, 1) * size(b, 1))

LinearAlgebra.det(::Id{T}) where T = one(T)
LinearAlgebra.logdet(::Id{T}) where T = log(one(T))

include("precompile.jl")

end # module IdentityMatrix

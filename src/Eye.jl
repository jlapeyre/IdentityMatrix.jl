# Methods for `Eye <: Diagonal` that override methods for `Diagonal`
using LinearAlgebra
export identitymatrix, materialize

import Base: inv, permutedims

# The fallback method for AbstractArray is already efficient for Idents for:
# length, firstindex, lastindex, axes, checkbounds

# The fallback method (for Diagonal) is already efficient for Eye for:
# getindex, factorize, real, float, ishermitian, issymmetric,
# isdiag, istriu, istril, log

function Base.iterate(iter::Idents{T}, istate = (1, 1)) where T
    (i::Int, j::Int) = istate
    m = size(iter, 1)
    return i > m ? nothing :
        ((@inbounds getindex(iter, i, j)),
         j == m ? (i + 1, 1) : (i, j + 1))
end

# The default method for Diagonal returns Diagonal with a dense diagonal
Base.imag(IM::Idents{T}) where T = Diagonal(Fill(zero(T), size(IM, 1)))
Base.iszero(::Idents) = false
Base.isone(::Idents) = true
LinearAlgebra.isposdef(::Idents) = true
LinearAlgebra.diag(IM::Idents{T}) where T = ones(T, size(IM, 1))
Base.sum(IM::Idents) = convert(eltype(IM), size(IM, 1))
Base.prod(IM::Idents) = zero(eltype(IM))
Base.first(::Idents{T}) where T = one(T)
Base.last(::Idents{T}) where T = one(T)
Base.minimum(::Idents{T}) where T = zero(T)
Base.maximum(::Idents{T}) where T = one(T)
Base.extrema(IM::Idents) = (minimum(IM), maximum(IM)) # FIXME: implement extrema(IM, dims = dims)

import LinearAlgebra: triu, triu!, tril, tril!

for f in (:permutedims, :triu, :triu!, :tril, :tril!, :inv) # These override inefficient methods for Eye
    @eval ($f)(IM::Idents) = IM
end

# This agrees with diag for dense matrices in that any integer `k` is allowed.
# It disagrees with diag for Diagonal, which throws and error when `abs(k)` is too large.
# But, there is probably a good reason for the difference.
function LinearAlgebra.diag(IM::Idents{T}, k::Integer) where T
    k == 0 && return LinearAlgebra.diag(IM)
    m = size(IM, 1)
    (k > m || k < -m) && return T[]
    return zeros(T, m - abs(k))
end

LinearAlgebra.det(::Idents{T}) where T = one(T)
LinearAlgebra.logdet(::Idents{T}) where T = log(one(T))
Base.sqrt(IM::Idents) = IM

# There are a few trig functions `f` for which `f(IM)` cannot be computed
# at compile time. This is one way to solve this problem.
const _cschval = csch(1.0)
mycsch(::Union{Type{Float64}, Type{Int}}) = _cschval
mycsch(::Type{T}) where T = csch(one(T))
Base.csch(IM::Idents) = LinearAlgebra.Diagonal(Fill(mycsch(eltype(IM)), size(IM, 1)))

Base.:(^)(IM::Idents, p::Integer) = IM

(Base.:/)(AM::AbstractMatrix, IM::Idents) = IM * AM
(Base.:/)(AM::Idents, IM::Idents) = IM * AM
(Base.:/)(AM::DenseMatrix, IM::Idents) = IM * AM

function (Base.:*)(IM::Idents{T}, AV::AbstractVector{V}) where {T, V}
    junk = checkuniquedim(IM, AV)
    return convert(Vector{Base.promote_op(*, T, V)}, AV)
end

function (Base.:*)(IM::Idents{T}, AM::AbstractMatrix{V}) where {T,V}
    junk = checkuniquedim(IM, AM)
    return convert(Matrix{Base.promote_op(*, T, V)}, AM)
end
(Base.:*)(AM::AbstractMatrix{T}, IM::Idents{V}) where {T, V} = IM * AM
(Base.:*)(AM::Diagonal, IM::Idents) = IM * AM
(Base.:*)(IMa::Eye{T}, IMb::Eye{V}) where {T, V} = Eye{Base.promote_op(*, T, V)}(size(IMa, 1))

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

####

## materializng is a mystery

# No. Today it is not faster. It is much slower
# This is faster than Matrix(I, n, n)
function identitymatrix(T::DataType, n::Int)
    a = zeros(T, n, n)
    @inbounds for i in 1:n # @inbounds does not change benchmark times
        a[i, i] = 1  # A bit faster than linear indexing
    end
    return a
end
identitymatrix(n::Integer) = identitymatrix(Float64, n)

# # The following method is ~7 times slower than the one below it.
# # This is baffling. There should be no difference
# # materialize(IM::Idents) = identitymatrix(eltype(IM), size(IM, 1))

# Some days, this is a bit faster than Matrix{T}(I, n, n)
function materialize(IM::Idents)
    a = zeros(eltype(IM), size(IM))
    @inbounds for i in 1:size(IM,1)  # @inbounds does not change benchmark times
        a[i, i] = 1  # A bit faster than linear indexing
    end
    return a
end

# For Eye{T}, the fallback method only materializes the diagonal.
# Here we create a dense matrix in agreement with `copymutable` for other `AbstractArray`
#Base.copymutable(IM::Idents) = Matrix{eltype(IM)}(I, size(IM))
Base.copymutable(IM::Idents) = materialize(IM)
#Base.copymutable(IM::Idents) = identitymatrix(eltype(IM), size(IM, 1))

####

ensuretype(::Type{T}, AM::AbstractArray{T}) where {T} =  AM
# FIXME: .* is slower than convert
ensuretype(::Type{T}, AM::AbstractArray{V, N}) where {T, V, N} = convert(Array{T, N}, AM)

# Put these last. The backslash confuses emacs.
# Diagonal is already efficient. But, we use `Idents` to remove fatal method ambiguity intrduced
# by the methods below.
(Base.:\)(IMa::Idents{T}, IMb::Idents{V}) where {T, V} = IMa * IMb
(Base.:\)(AM::AbstractMatrix{T}, IM::Idents{V}) where {T, V} = ensuretype(Base.promote_op(*, T, V), Base.inv(AM))
(Base.:\)(IM::Idents{V}, AM::AbstractMatrix{T}) where {T, V} = ensuretype(Base.promote_op(*, T, V), AM)

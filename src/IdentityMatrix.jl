"""
    module IdentityMatrix

Methods for `FillArrays.Eye`.
"""
module IdentityMatrix

using LinearAlgebra, FillArrays
import LinearAlgebra: triu, triu!, tril, tril!
import Base: inv, permutedims

export identitymatrix, materialize

# FIXME: replace checkuniquedim with something more efficient (from LinearAlgebra or Base)
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

# The fallback method for AbstractArray is already efficient for Eye for:
# length, firstindex, lastindex, axes, checkbounds

# The fallback method (for Diagonal) is already efficient for Eye for:
# getindex, factorize, real, float, ishermitian, issymmetric, isdiag, istriu,
# istril, log, most trig functions

function Base.iterate(iter::Eye{T}, istate = (1, 1)) where T
    (i::Int, j::Int) = istate
    m = size(iter, 1)
    return i > m ? nothing :
        ((@inbounds getindex(iter, i, j)),
         j == m ? (i + 1, 1) : (i, j + 1))
end

# The default method for Diagonal returns Diagonal with a dense diagonal
Base.imag(IM::Eye{T}) where T = Diagonal(Fill(zero(T), size(IM, 1)))
Base.iszero(::Eye) = false
Base.isone(::Eye) = true
LinearAlgebra.isposdef(::Eye) = true
LinearAlgebra.diag(IM::Eye{T}) where T = ones(T, size(IM, 1))
Base.sum(IM::Eye) = convert(eltype(IM), size(IM, 1))
Base.prod(IM::Eye) = zero(eltype(IM))
Base.first(::Eye{T}) where T = one(T)
Base.last(::Eye{T}) where T = one(T)
Base.minimum(::Eye{T}) where T = zero(T)
Base.maximum(::Eye{T}) where T = one(T)
Base.extrema(IM::Eye) = (minimum(IM), maximum(IM)) # FIXME: implement extrema(IM, dims = dims)

for f in (:permutedims, :triu, :triu!, :tril, :tril!, :inv) # These override inefficient methods for Eye
    @eval ($f)(IM::Eye) = IM
end

# This agrees with diag for dense matrices in that any integer `k` is allowed.
# It disagrees with diag for Diagonal, which throws an error when `abs(k)` is too large.
function LinearAlgebra.diag(IM::Eye{T}, k::Integer) where T
    k == 0 && return LinearAlgebra.diag(IM)
    m = size(IM, 1)
    (k > m || k < -m) && return T[]
    return zeros(T, m - abs(k))
end

LinearAlgebra.det(::Eye{T}) where T = one(T)
LinearAlgebra.logdet(::Eye{T}) where T = log(one(T))
Base.sqrt(IM::Eye) = IM

# There are a few trig functions `f` for which `f(IM)` cannot be computed
# at compile time. This is one way to solve this problem.
const _cschval = csch(1.0)
_mycsch(::Union{Type{Float64}, Type{Int}}) = _cschval
_mycsch(::Type{T}) where T = csch(one(T))
Base.csch(IM::Eye) = LinearAlgebra.Diagonal(Fill(_mycsch(eltype(IM)), size(IM, 1)))

Base.:(^)(IM::Eye, p::Integer) = IM

(Base.:/)(AM::AbstractMatrix, IM::Eye) = IM * AM
(Base.:/)(AM::Eye, IM::Eye) = IM * AM
(Base.:/)(AM::DenseMatrix, IM::Eye) = IM * AM

function (Base.:*)(IM::Eye{T}, AV::AbstractVector{V}) where {T, V}
    junk = checkuniquedim(IM, AV)
    return convert(Vector{Base.promote_op(*, T, V)}, AV)
end

function (Base.:*)(IM::Eye{T}, AM::AbstractMatrix{V}) where {T,V}
    junk = checkuniquedim(IM, AM)
    return convert(Matrix{Base.promote_op(*, T, V)}, AM)
end
(Base.:*)(AM::AbstractMatrix{T}, IM::Eye{V}) where {T, V} = IM * AM
(Base.:*)(AM::Diagonal, IM::Eye) = IM * AM
(Base.:*)(IMa::Eye{T}, IMb::Eye{V}) where {T, V} = Eye{Base.promote_op(*, T, V)}(size(IMa, 1))

function Base.kron(a::Eye{T}, b::AbstractMatrix{S}) where {T, S}
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

function Base.kron(a::AbstractMatrix{T}, b::Eye{S}) where {T, S}
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

Base.kron(a::Eye{T}, b::Eye{T}) where T = Eye{T}(size(a, 1) * size(b, 1))
Base.kron(a::Eye{T}, b::Eye{V}) where {T, V} = Eye{Base.promote_op(*, T, V)}((size(a, 1) * size(b, 1)))

LinearAlgebra.eigvals(IM::Eye{T}) where T = diag(IM)
LinearAlgebra.eigvecs(IM::Eye) = IM # method for Diagonal returns a material matrix
LinearAlgebra.eigen(IM::Eye) = LinearAlgebra.Eigen(LinearAlgebra.eigvals(IM), LinearAlgebra.eigvecs(IM))

function identitymatrix(::Type{T}, n::Int) where T
    a = zeros(T, n, n)
    @inbounds for i in 1:n
        a[i, i] = 1
    end
    return a
end
identitymatrix(n::Integer) = identitymatrix(Float64, n)

materialize(IM::Eye) = identitymatrix(eltype(IM), size(IM, 1))
# materialize(IM::Eye) = Matrix{eltype(IM)}(I, size(IM))

# For Eye{T}, the fallback method only materializes the diagonal.
# Here we create a dense matrix in agreement with `copymutable` for other `AbstractArray`.
Base.copymutable(IM::Eye) = materialize(IM)
# Matrix always makes a full copy
Base.Matrix(IM::Eye) = materialize(IM)

ensuretype(::Type{T}, AM::AbstractArray{T}) where {T} =  AM
# FIXME: .* is slower than convert
ensuretype(::Type{T}, AM::AbstractArray{V, N}) where {T, V, N} = convert(Array{T, N}, AM)

# Put these last. The backslash confuses emacs.
# Diagonal is already efficient. But, we use `Eye` to remove fatal method ambiguity intrduced
# by the methods below.
(Base.:\)(IMa::Eye{T}, IMb::Eye{V}) where {T, V} = IMa * IMb
(Base.:\)(AM::AbstractMatrix{T}, IM::Eye{V}) where {T, V} = ensuretype(Base.promote_op(*, T, V), Base.inv(AM))
(Base.:\)(IM::Eye{V}, AM::AbstractMatrix{T}) where {T, V} = ensuretype(Base.promote_op(*, T, V), AM)

end # module

"""
    module IdentityMatrix

Methods for `Diagonal`, `FillArrays.Fill`, and `FillArrays.Eye`.
"""
module IdentityMatrix

using LinearAlgebra, FillArrays
import LinearAlgebra: triu, triu!, tril, tril!
import LinearAlgebra: norm, normp, norm1, norm1, normInf, normMinusInf
import Base: inv, permutedims
import StatsBase

export idmat, materialize
export norm

checkempty(x) = isempty(x) ? throw(ArgumentError("Got empty container.")) : return nothing
checkzerodim(x) = (m = size(x, 1); return m == 0 ? throw(ArgumentError("Got empty container.")) : m)

# FIXME: replace checkuniquedim with something more efficient (from LinearAlgebra or Base)
checkuniquedim(D::LinearAlgebra.Diagonal) = size(D, 1)
checkuniquedim(AV::AbstractVector) = size(AV, 1)

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

# Following are functions that I determined will *not* benefit from
# a specialized method.
# The fallback method for (to AbstractArray) is already efficient for Eye for:
# length, firstindex, lastindex, axes, checkbounds

# The fallback method (to Diagonal) is already efficient for Eye for:
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
Base.one(IM::Eye) = IM
Base.oneunit(IM::Eye) = Base.one(IM)
Base.zero(IM::Eye{T}) where T = Diagonal(Zeros{T}(size(IM, 1)))
LinearAlgebra.isposdef(::Eye) = true

# Return a Vector to agree with other `diag` methods
LinearAlgebra.diag(IM::Eye{T}) where T = ones(T, size(IM, 1))
Base.sum(f::Function, IM::Eye{T}) where T = (m = size(IM, 1); (m * (m - 1)) * f(zero(T)) + m * f(one(T)))
Base.sum(IM::Eye{T}) where T = convert(T, size(IM, 1))
Base.prod(f, IM::Eye{T}) where T = (m = size(IM, 1); f(zero(T))^(m * (m - 1)) * f(one(T))^m)
Base.prod(IM::Eye{T}) where T = size(IM, 1) > 1 ? zero(T) : one(T)
# Error checking is done in `norm`, which calls the following

norm2(IM::Eye{T}) where T = sqrt(T(size(IM, 1)))
norm1(IM::Eye{T}) where T = T(size(IM, 1))
normInf(IM::Eye{T}) where T = one(T)
normMinusInf(IM::Eye{T}) where T = zero(T)

normp(IM::Eye{T}, p::Real) where T = (m = size(IM, 1); return iszero(p) ? float(T(size(IM, 1))) : T(m)^(1/p))

# We have to reproduce this, because LinearAlgebra does not split norm0 into a function.
# So, we are unable to replace it.
# We could file an issue
function norm(IM::Eye{T}, p::Real=2) where T
    isempty(IM) && return float(zero(T))
    if p == 2
        return norm2(IM)
    elseif p == 1
        return norm1(IM)
    elseif p == Inf
        return normInf(IM)
    elseif p == 0
        return float(T(size(IM, 1)))
    elseif p == -Inf
        return normMinusInf(IM)
    else
        return normp(IM, p)
    end
end

LinearAlgebra.opnorm(IM::Eye{T}, p::Real=1) where T = (isempty(IM) ? zero(T) : one(T)) |> float

Base.first(::Eye{T}) where T = one(T)
Base.last(::Eye{T}) where T = one(T)

function Base.minimum(IM::Eye{T}) where T
    m = size(IM, 1)
    m > 1 && return zero(T)
    m < 1 && return Base.minimum(T[]) # Error
    return one(T)
end

function Base.maximum(IM::Eye{T}) where T
    size(IM, 1) == 0 && return Base.maximum(T[])
    return one(T)
end

function Base.minimum(D::Diagonal{T}) where T
    mindiag = Base.minimum(D.diag)
    size(D, 1) > 1 && return (min(zero(T),mindiag))
    return mindiag
end

Base.extrema(IM::Eye) = (minimum(IM), maximum(IM)) # FIXME: implement extrema(IM, dims = dims)

# StatsBase.mean is sum/length and therefore is efficient.
#StatsBase.mean(IM::Eye{T}) where T = (m = size(IM, 1); convert(T, m * (m - 1)))
function StatsBase.median(IM::Eye{T}) where T
    m = checkzerodim(IM)
    m == 1 && return float(T(1))
    m == 2 && return float(T(1//2))
    return float(zero(T))
end

# const Callable = Union{Function, DataType}
# Can't use Callable maybe :( Method ambiguity
Base.any(f::Function, IM::Eye{T}) where T = f(zero(T)) || f(one(T))
Base.all(f::Function, IM::Eye{T}) where T = f(zero(T)) && f(one(T))

for f in (:permutedims, :triu, :triu!, :tril, :tril!, :inv)
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

(Base.:^)(IM::Eye, p::Integer) = IM
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

# Kron with first argment either Diagonal or Eye
for (diagonaltype, A_jj, outputelement) in ((:(A::Diagonal{T}), :(A_jj = A[j, j]), :(A_jj * B[l,k]) ),
                        ( :(A::Eye{T}), :(nothing), :(B[l,k])))
    @eval begin
        function (Base.kron)($diagonaltype, B::AbstractMatrix{S}) where {T<:Number, S<:Number}
            @assert ! Base.has_offset_axes(B)
            (mA, nA) = size(A); (mB, nB) = size(B)
            R = zeros(Base.promote_op(*, T, S), mA * mB, nA * nB)
            m = 1
            for j = 1:nA
                $A_jj
                for k = 1:nB
                    for l = 1:mB
                        R[m] = $outputelement
                        m += 1
                    end
                    m += (nA - 1) * mB
                end
                m += mB
            end
            return R
        end
    end
end

function Base.kron(A::AbstractMatrix{T}, B::Diagonal{S}) where {T<:Number, S<:Number}
    @assert ! Base.has_offset_axes(A)
    (mA, nA) = size(A); (mB, nB) = size(B)
    R = zeros(Base.promote_op(*, T, S), mA * mB, nA * nB)
    m = 1
    for j = 1:nA
        for l = 1:mB
            Bll = B[l,l]
            for k = 1:mA
                R[m] = A[k,j] * Bll
                m += nB
            end
            m += 1
        end
        m -= nB
    end
    return R
end

# It is not clear if there is any advantage over kron(::AbstractMatrix, ::Diagonal)
function Base.kron(A::AbstractMatrix{T}, B::Eye{S}) where {T<:Number, S<:Number}
    @assert ! Base.has_offset_axes(A)
    (mA, nA) = size(A); (mB, nB) = size(B)
    R = zeros(Base.promote_op(*, T, S), mA * mB, nA * nB)
    m = 1
    for j = 1:nA
        for l = 1:mB
            for k = 1:mA
                R[m] = A[k,j]
                m += nB
            end
            m += 1
        end
        m -= nB
    end
    return R
end

Base.kron(a::Eye{T}, b::Eye{T}) where {T<:Number} = Eye{T}(size(a, 1) * size(b, 1))
Base.kron(a::Eye{T}, b::Eye{V}) where {T<:Number, V<:Number} = Eye{Base.promote_op(*, T, V)}((size(a, 1) * size(b, 1)))

LinearAlgebra.eigvals(IM::Eye{T}) where T = diag(IM)
LinearAlgebra.eigvecs(IM::Eye) = IM # method for Diagonal returns a material matrix
LinearAlgebra.eigen(IM::Eye) = LinearAlgebra.Eigen(LinearAlgebra.eigvals(IM), LinearAlgebra.eigvecs(IM))

"""
    idmat(::Type{T}, n::Int) where T
    idmat(n::Integer)

Create an identity matrix of type `Matrix{T}`. The default for
`T` is `Float64`.
"""
function idmat(::Type{T}, n::Integer) where T
    a = zeros(T, n, n)
    for i in 1:n
        a[i, i] = 1
    end
    return a
end
idmat(n::Integer) = idmat(Float64, n)

materialize(IM::Eye) = idmat(eltype(IM), size(IM, 1))
# materialize(IM::Eye) = Matrix{eltype(IM)}(I, size(IM)) # This is a bit slower
Base.Matrix(IM::Eye) = materialize(IM)

# For Eye{T}, the fallback method only materializes the diagonal.
Base.copymutable(IM::Eye) = Diagonal(ones(eltype(IM), size(IM, 1)))

# Diagonal in general
Broadcast.broadcasted(f::T, x::Number, D::Diagonal{<:Number}) where T <: typeof(*) = Diagonal(broadcast(f, x, D.diag))
Broadcast.broadcasted(f::T, D::Diagonal{<:Number}, x::Number) where T <: typeof(*) = broadcast(f, x, D)

Base.:+(IM::Eye{T}, s::UniformScaling) where T = Diagonal(Fill(one(T) + s.λ, size(IM, 1)))
Base.:+(s::UniformScaling, IM::Eye) = IM + s
Base.:-(IM::Eye{T}, s::UniformScaling) where T = Diagonal(Fill(one(T) - s.λ, size(IM, 1)))
Base.:-(s::UniformScaling, IM::Eye{T}) where T = Diagonal(Fill(s.λ - one(T), size(IM, 1)))

# Put these last. The backslash confuses emacs.
# Diagonal is already efficient. But, we use `Eye` to remove fatal method ambiguity intrduced
# by the methods below.
(Base.:\)(IMa::Eye{T}, IMb::Eye{V}) where {T, V} = IMa * IMb
(Base.:\)(AM::AbstractMatrix{T}, IM::Eye{V}) where {T, V} = convert(AbstractMatrix{Base.promote_op(*, T, V)}, inv(AM))
(Base.:\)(IM::Eye{V}, AM::AbstractMatrix{T}) where {T, V} = convert(AbstractMatrix{Base.promote_op(*, T, V)}, AM)

(Base.:\)(IM::Eye, s::UniformScaling) = s.λ * IM
(Base.:\)(s::UniformScaling, IM::Eye) =  IM / s.λ

end # module

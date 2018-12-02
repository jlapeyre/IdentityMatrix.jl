# This code changes behavior in LinearAlgebra.Diagonal
# It can't go to FillArrays

function Base.sum(f::Function, D::LinearAlgebra.Diagonal{T}) where T
    m = size(D, 1)
    return (m * (m - 1)) * f(zero(T)) + sum(f, D.diag)
end
Base.sum(D::LinearAlgebra.Diagonal) = sum(identity, D)

function Base.prod(f::Function, D::LinearAlgebra.Diagonal{T}) where T
    m = size(D, 1)
    return f(zero(T))^(m * (m - 1)) * prod(f, D.diag)^m
end
Base.prod(D::LinearAlgebra.Diagonal) = prod(identity, D)

function Base.minimum(D::Diagonal{T}) where T <: Number
    mindiag = Base.minimum(D.diag)
    size(D, 1) > 1 && return (min(zero(T), mindiag))
    return mindiag
end

function Base.maximum(D::Diagonal{T}) where T <: Number
    maxdiag = Base.maximum(D.diag)
    size(D, 1) > 1 && return (max(zero(T), maxdiag))
    return maxdiag
end

Base.any(f::Function, x::Diagonal{T}) where T <: Number = size(x, 1) == 1 ? f(x[1]) :
    f(zero(T)) || any(f, x.diag)

Base.all(f::Function, x::Diagonal{T}) where T <: Number = size(x, 1) == 1 ? f(x[1]) :
    f(zero(T)) && all(f, x.diag)

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

# Broadcast.broadcasted(f::typeof(*), x::Number, D::Diagonal{<:Number}) = Diagonal(Broadcast.broadcast(f, x, D.diag))
# Broadcast.broadcasted(f::typeof(*), D::Diagonal{<:Number}, x::Number) = Broadcast.broadcasted(f, x, D)

# for ft in (round, floor, abs, sqrt, sin, sind, cbrt, tan)
#     @eval Broadcast.broadcasted(f::typeof($ft), D::Diagonal{<:Number}) = Diagonal(broadcast(f, D.diag))
# end

function Broadcast.broadcasted(f::Function, D::Diagonal{T}) where T <: Number
    fz = f(zero(T))
    return iszero(fz) ? broadcasted_diag_zero(f, D) : broadcasted_diag_full(f, fz, D)
end

@inline function broadcasted_diag_zero(f, D)
    return Diagonal(broadcast(f, D.diag))
end

@inline function broadcasted_diag_full(f, fz, D::Diagonal{T}) where T
    R = fill(fz, size(D))
    for i in 1:size(D, 1)
        R[i,i] = f(D.diag[i])
    end
    return R
end





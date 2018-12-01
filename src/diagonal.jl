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

function Base.minimum(D::Diagonal{T}) where T
    mindiag = Base.minimum(D.diag)
    size(D, 1) > 1 && return (min(zero(T),mindiag))
    return mindiag
end

Base.any(f::Function, x::Diagonal{T}) where T = size(x, 1) == 1 ? f(x[1]) :
    f(zero(T)) || any(f, x.diag)

Base.all(f::Function, x::Diagonal{T}) where T = size(x, 1) == 1 ? f(x[1]) :
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

Broadcast.broadcasted(f::T, x::Number, D::Diagonal{<:Number}) where T <: typeof(*) = Diagonal(broadcast(f, x, D.diag))
Broadcast.broadcasted(f::T, D::Diagonal{<:Number}, x::Number) where T <: typeof(*) = broadcast(f, x, D)

Base.:+(IM::Eye{T}, s::UniformScaling) where T = Diagonal(Fill(one(T) + s.λ, size(IM, 1)))
Base.:+(s::UniformScaling, IM::Eye) = IM + s
Base.:-(IM::Eye{T}, s::UniformScaling) where T = Diagonal(Fill(one(T) - s.λ, size(IM, 1)))
Base.:-(s::UniformScaling, IM::Eye{T}) where T = Diagonal(Fill(s.λ - one(T), size(IM, 1)))
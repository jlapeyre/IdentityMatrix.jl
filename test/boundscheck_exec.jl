using IdentityMatrix
using Test

# These are in a separte file so that they can be run with boundscheck set to auto.

function mysum_inbounds(m)
    s = zero(eltype(m))
    @inbounds for i in 1:10
        s += m[i, i]
    end
    s
end

function mysum(m)
    s = zero(eltype(m))
    for i in 1:10
        s += m[i, i]
    end
    s
end

@testset "Bounds check" begin
    @test mysum_inbounds(Id(3)) == 10.0
    @test_throws BoundsError mysum(Id(3))
end

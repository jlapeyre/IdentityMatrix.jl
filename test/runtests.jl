using IdentityMatrix
using Test
import LinearAlgebra
using MethodInSrc

@testset "Methods in source" begin
    for T in (Int, Float64, ComplexF64)
        for N in (1, 2, 4, 10)
            mrand = rand(N, N)
            M = Id(T, N)
            @test @isinsrc isreal(M)
            @test @isinsrc in(3, M)
            @test @isinsrc isone(M)
            @test @isinsrc iszero(M)
            @test @isinsrc LinearAlgebra.diag(M)
            @test @isinsrc LinearAlgebra.diag(M, 2)
            @test @isinsrc LinearAlgebra.eigvals(M)
            @test @isinsrc LinearAlgebra.isposdef(M)
            @test @isinsrc size(M)
            @test @isinsrc complex(M)
            @test @isinsrc float(M)
            @test @isinsrc big(M)
            @test @isinsrc real(M)
            @test @isinsrc M * mrand
            @test @isinsrc mrand * M
            @test @isinsrc 3.0 * M
            @test @isinsrc M * 3.1
        end
    end
end

@testset "IdentityMatrix.jl" begin
    @test eachindex(Id(3)) == CartesianIndices((3, 3))
    @test size(Id(5)) == (5, 5)

    @test Id(2) == [1 0; 0 1]

    for n in (1, 2, 3, 4, 10, 100)
        @test Id(n) == LinearAlgebra.I(n)
        @test Id(n) == float(LinearAlgebra.I(n))

        @test Id(Int, n) == LinearAlgebra.I(n)
        @test Id(Int, n) == float(LinearAlgebra.I(n))

        @test Id(Bool, n) == LinearAlgebra.I(n)
        @test Id(Bool, n) == float(LinearAlgebra.I(n))
    end

    @test Id(3)[1, 1] == 1
    @test Id(3)[2, 3] == 0
    @test Id(3) isa Id{Float64, 3}
    @test Id(Int, 4) isa Id{Int, 4}
    @test_throws MethodError Id(3.1)
    @test_throws MethodError Id(3.0)
    @test_throws MethodError Id(String, 2)[1, 2]

    @test Id(Int, 4) === Id(Int, 4)
    m = Id(Int, 4)
    @test m^2 === m
    @test m^10 === m
    @test m^-3 === m

    @test prod(Id(3)) === 0.0
    @test prod(Id(1)) === 1.0
    @test prod(Id(Int, 3)) === 0
    @test prod(Id(Int, 1)) === 1
    @test sum(Id(3)) === 3.0
    @test sum(Int, Id(3)) === 3
    @test one(Id(10)) === Id(10)

    @test isone(Id(4))
    @test ! iszero(Id(4))

    @test real(complex(Id(2))) isa Id{Float64, 2}
    @test complex(Id(2)) isa Id{ComplexF64, 2}
    @test complex(Id(Float32, 2)) isa Id{ComplexF32, 2}
    @test big(Id(2)) isa Id{BigFloat, 2}
    @test big(Id(Int, 2)) isa Id{BigInt, 2}

    @test LinearAlgebra.eigvals(Id(2)) == ones(Float64, 2)

    if ! (VERSION < v"1.7")
        @test ! ismutabletype(Id{T, N} where {T, N})
    end
end

# Bounds check tests must run in in a different process because bounds checking is enabled
# when running the test suite.
let cmd = `$(Base.julia_cmd()) --check-bounds=auto --depwarn=error --startup-file=no boundscheck_exec.jl`
    success(pipeline(cmd; stdout=stdout, stderr=stderr)) || error("boundscheck test failed, cmd : $cmd")
end

# let cmd = `$(Base.julia_cmd()) --check-bounds=no --depwarn=error --startup-file=no boundscheck_exec.jl`
#     success(pipeline(cmd; stdout=stdout, stderr=stderr)) || error("boundscheck test failed, cmd : $cmd")
# end

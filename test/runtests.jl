using IdentityMatrix
using Test
import LinearAlgebra
using LinearAlgebra: eigvals, eigen, eigmin, eigmax,
    Diagonal, kron
using MethodInSrc

# Some of these are included in specific sections
@testset "Methods in source" begin
    for T in (Int, Float64, ComplexF64)
        for N in (1, 2, 4, 10)
            mrand = rand(N, N)
            M = Id(T, N)
            @test @insrc isreal(M)
            @test @insrc in(1, M)
            @test @insrc isone(M)

            @test @isinsrc iszero(M)
            @test @isinsrc LinearAlgebra.diag(M)
            @test @isinsrc LinearAlgebra.diag(M, 2)
            @test @isinsrc LinearAlgebra.eigvals(M)
            @test @isinsrc LinearAlgebra.isposdef(M)
            @test @isinsrc LinearAlgebra.ishermitian(M)
            @test @isinsrc LinearAlgebra.issymmetric(M)
            @test @isinsrc size(M)
            @test @isinsrc complex(M)
            @test @isinsrc float(M)
            @test @isinsrc big(M)
            @test @isinsrc real(M)
            @test @isinsrc M * mrand
            @test @isinsrc mrand * M
            @test @isinsrc 3.0 * M
            @test @isinsrc M * 3.1
            @test @isinsrc collect(M)
#            @test @isinsrc M == Id(T, 3)

            @test @isinsrc copyto!(mrand, M)

            @test ! @isinsrc eltype(M)
            @test ! @isinsrc length(M)
        end
    end
end

@testset "Construction" begin
    id = Id{ComplexF64, 3}()
    @test eltype(id) == ComplexF64
    @test isreal(id)
    @test Id(Complex, 3) == Id{ComplexF64, 3}()
end

@testset "Properties" begin
    @test LinearAlgebra.isposdef(Id(3))
    @test LinearAlgebra.ishermitian(Id(3))
    @test LinearAlgebra.issymmetric(Id(3))
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

    @test string(Id(3)) == "Id{Float64, 3}()"
    @test string(Id(ComplexF64, 10)) == "Id{ComplexF64, 10}()"
    io = IOBuffer()
    show(io, MIME"text/plain"(), Id(3))
    @test String(take!(io)) == string(Id(3))
    @test copy(Id(4)) === Id(4)

    idm = collect(Id(2))
    @test isa(idm, Matrix{Float64})
    @test size(idm) == (2, 2)
    @test [Id(2)[i] for i in 1:4] == [1.0, 0.0, 0.0, 1.0]

    mrand = rand(2, 2)
    result = copyto!(mrand, Id(2))
    @test result === mrand
    @test mrand == Id(2)
    @test eltype(mrand) == Float64

    if ! (VERSION < v"1.7")
        @test ! ismutabletype(Id{T, N} where {T, N})
    end
end

@testset "multiplication" begin
    for T in (Float64, ComplexF64, Int8, UInt)
        @test Id(3) * Id(3) == Id(3)
        @test eltype(Id(3) * Id(3)) == eltype(Id(3))
    end

    @test eltype(Id(Float64, 3) * Id(Int, 3)) == Float64
    @test_throws DimensionMismatch Id(3) * Id(2)
    @test_throws DimensionMismatch Id(Int, 3) * Id(Float64, 2)

    mrand = rand(2, 2)
    @test Id(2) * mrand == mrand
    @test mrand * Id(2) == mrand
    @test mrand * Id(Complex, 2) == complex(mrand)
    @test Id(Complex, 2) * mrand == complex(mrand)

    m_diag = LinearAlgebra.diagm([1, 2, 3])
    fm_diag = float(m_diag)
    @test m_diag * Id(3) == fm_diag
    @test Id(3) * m_diag == fm_diag
    @test fm_diag * Id(Int, 3) == fm_diag
    @test Id(Int, 3) * fm_diag == fm_diag

    m_scaled_expected = LinearAlgebra.UniformScaling(13)(3)
    m_scaled = 13 * Id(Int, 3)
    @test m_scaled == m_scaled_expected
    @test eltype(m_scaled) == Int
    @test isa(m_scaled, LinearAlgebra.Diagonal)

    fm_scaled_expected = float(m_scaled_expected)
    fm_scaled = 13 * Id(3)
    @test fm_scaled == m_scaled_expected
    @test eltype(fm_scaled) == Float64
    @test eltype(fm_scaled) == eltype(fm_scaled_expected)
    @test isa(fm_scaled, LinearAlgebra.Diagonal)

    mdiag = Diagonal([1,2,3])
    mult_result = Id(Int, 3) * Diagonal([1,2,3])
    @test mult_result == mdiag
    @test typeof(mdiag) == typeof(mult_result)

    mult_result = Id(Float64, 3) * Diagonal([1,2,3])
    @test mult_result == mdiag
    @test mult_result isa Diagonal{Float64, Vector{Float64}}
end

@testset "Linear Algebra" begin
    for T in (Int, Float64, ComplexF64)
        evals = LinearAlgebra.eigvals(Id(T, 2))
        @test evals == ones(2)
        @test evals isa Vector{T}
        im = Id(T,3)
        @test eigmin(im) == 1
        @test eigmax(im) == 1
        @test eigmin(im) isa T
        @test eigmax(im) isa T
        @test eigen(im) == eigen(collect(im))
    end
    @test kron(Id(3), Id(4)) == Id(12)
    @test kron(Id(3), Id(Int, 4)) == Id(12)
end

@testset "Comparisons" begin
    for T in (Int, Float64, ComplexF64)
        M = Id(T, 3)
        @test @isinsrc M == Id(T, 3)
        @test @isinsrc M == Id(3)
        @test Base.:(==)(M, Id(T, 3))
        @test M == Id(3)
    end
end

@testset "Predicates" begin
    for T in (Int, Float64, ComplexF64)
        M = Id(T, 3)
        @test @isinsrc isone(M)
        @test @isinsrc iszero(M)
        @test isone(M)
        @test !iszero(M)
    end
    @test_throws MethodError isone(Id(String, 3))
end

# Bounds check tests must run in in a different process because bounds checking is enabled
# when running the test suite.
let cmd = `$(Base.julia_cmd()) --check-bounds=auto --depwarn=error --startup-file=no boundscheck_exec.jl`
    success(pipeline(cmd; stdout=stdout, stderr=stderr)) || error("boundscheck test failed, cmd : $cmd")
end

# let cmd = `$(Base.julia_cmd()) --check-bounds=no --depwarn=error --startup-file=no boundscheck_exec.jl`
#     success(pipeline(cmd; stdout=stdout, stderr=stderr)) || error("boundscheck test failed, cmd : $cmd")
# end

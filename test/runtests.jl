using IdentityMatrix
using Test
using LinearAlgebra
using FillArrays

@testset "mul and div" begin
    ncols = 4
    IM = eye(ncols)
    @test IM * IM == IM
    @test IM / IM == IM
    @test IM \ IM == IM

    rm = rand(ncols, ncols)
    @test IM * rm == rm
    @test rm * IM == rm
    @test IM / rm == inv(rm)
    @test rm / IM == rm
    @test rm \ IM == inv(rm)
end

@testset "constructors" begin
    ncols = 10
    @test eye(ncols) == Identity(ncols)
    @test eye(ncols) == Identity{Float64}(ncols)
    @test eye(Float64, ncols) == eye(ncols)
    @test eye(Int, ncols) == Identity{Int}(ncols)
    @test eye(zeros(2,2)) == eye(2)
    @test eye(Int, zeros(2,2)) == eye(Int, 2)
    @test eye(Int, eye(2)) == Identity{Int}(2)
end

@testset "conversion" begin
    ncols = 10
    m = eye(ncols)
    @test Identity(m) === m
    @test AbstractMatrix{Int}(m) == eye(Int, m)
    @test Identity(ncols) == m
    mint = eye(Int, ncols)
    @test m == mint
    md = Matrix(m)
    @test md == Matrix(LinearAlgebra.I, ncols, ncols)
    @test eltype(m) == Float64
    @test eltype(mint) == Int

    @test eye(ncols) == eye(Diagonal(ones(ncols)))
    @test eye(ncols) == Diagonal(ones(ncols))
    @test diag(eye(ncols)) == ones(ncols)
end

@testset "reductions" begin
    ncols = 10
    m = eye(ncols)
    @test sum(m) == ncols
    @test prod(m) == 0
    for T in (Int, Float64, Int32, BigInt, Rational{Int})
        for fop in (sum, prod)
            im = eye(T, ncols)
            r = fop(im)
            @test typeof(r) == eltype(im)
        end
    end
end

@testset "transform" begin
    m = eye(3)
    for f in (transpose, adjoint, conj, triu, triu!, tril, tril!, inv)
        @test f(m) == m # FIXME: why does === fail ?
    end
end

@testset "kron" begin
    ncols1 = 3
    ncols2 = 4
    for eyeT in (eye, Eye)
        im = eyeT(ncols1)
        imd = Matrix(im)
        md = rand(ncols2, ncols2)
        @test kron(imd, md) == kron(im, md)
        @test kron(md, imd) == kron(md, im)
        @test isone(kron(im, im))
        @test isa(kron(im, im), typeof(im))
    end
end

@testset "eigen" begin
    ncols = 5
    for eyeT in (eye, Eye)
        im = eyeT(ncols)
        md = Matrix(im)
        ri = eigen(im)
        rd = eigen(md)
        @test isapprox(ri.values, rd.values)
        @test isapprox(ri.vectors, rd.vectors)
        @test isapprox(eigvals(im), eigvals(md))
        @test isapprox(eigvecs(im), eigvecs(md))
        @test isa(ri.vectors, typeof(im))
    end
end

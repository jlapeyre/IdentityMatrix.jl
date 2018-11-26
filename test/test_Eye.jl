@testset "Eye" begin
    N = 5
    IM = Eye(N)
    IMd = Matrix(IM)
    @test isa(imag(IM), Diagonal)
    for f in (imag, iszero, isone, isposdef, sum, prod, first, last,
              minimum, maximum, extrema, triu, triu!, tril, tril!, inv,
              diag, det, logdet, sqrt)
        @test f(IM) == f(copy(IMd))
    end
    for p in -5:5
        @test IM^p == IM
    end
    Base.copymutable(IM) == ones(eltype(IM), size(IM))
end

@testset "iterate" begin
    N = 5
    IM = Eye(N)
    @test [x for x in IM] == IM
end

@testset "mul and div" begin
    ncols = 4
    IM = Eye(ncols)
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

@testset "reductions" begin
    ncols = 10
    m = Eye(ncols)
    @test sum(m) == ncols
    @test prod(m) == 0
    for T in (Int, Float64, Int32, BigInt, Rational{Int})
        for fop in (sum, prod)
            im = Eye{T}(ncols)
            r = fop(im)
            @test typeof(r) == eltype(im)
        end
    end
end

@testset "transform" begin
    m = Eye(3)
    for f in (triu, triu!, tril, tril!, inv)
        @test f(m) == m # FIXME: why does === fail ?
    end
end

@testset "kron" begin
    ncols1 = 3
    ncols2 = 4
    for eyeT in (Eye, )
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
    for eyeT in (Eye, )
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

using IdentityMatrix
using Test
using LinearAlgebra
using FillArrays
using MethodInSrc

@testset  "norms" begin
    for ncols in (0, 1, 3, 10)
        for T in (Float64, Int)
            m = Eye{T}(ncols)
            md = Matrix(m)
            for op in (opnorm, norm, x -> norm(x, 1), x -> norm(x, 2), x -> norm(x, Inf), x -> norm(x, -Inf), x -> norm(x, 3//2))
                @test op(m) == op(md)
            end
            @test @isinsrc opnorm(m)
        end
    end
end

@testset "Eye" begin
    for ncols in (1, 2, 5)
        IM = Eye(ncols)
        IMd = Matrix(IM)
        @test isa(imag(IM), Diagonal)
        for f in (imag, isposdef, sum, prod, first, last,
                  minimum, maximum, extrema, triu, triu!, tril, tril!, inv,
                  diag, det, logdet, sqrt)
            @test @insrc(f(IM)) == f(copy(IMd))
        end
        for f in (iszero, )
            @test @ninsrc(f(IM)) == f(copy(IMd))
        end
        for p in -5:5
            @test IM == @insrc IM^p
        end
        @test isa(copy(IM), Diagonal)
        @test isa(@insrc(Matrix(IM)), Matrix)
    end
end

@testset "mul and div" begin
    for ncols in (1, 3 ,10)
        IM = Eye(ncols)
        @test IM === @insrc IM * IM
        @test IM === @insrc IM / IM
        @test IM === @insrc IM \ IM

        rm = rand(ncols, ncols)
        @test rm === @insrc IM * rm
        @test rm === @insrc rm * IM
        @test rm === @insrc rm / IM
        @test inv(rm) == @insrc IM / rm
        @test inv(rm) == @insrc rm \ IM
        @test rm == @insrc IM \ rm
    end
end

@testset "algebra with AbstractFill" begin
    for d in (1, 2, 3, 10)
       for  T1 in (Int, Float64, ComplexF64)
           for T2 in (Int, Float64, ComplexF64)
               @test @insrc(Eye{T1}(d) * Zeros{T2}(d, d)) == Zeros{typeof(one(T1)*one(T2))}(d, d)
               @test @insrc(Eye{T1}(d) * Zeros{T2}(d)) == Zeros{typeof(one(T1)*one(T2))}(d)
               @test @insrc(Eye{T1}(d) * Ones{T2}(d, d)) == Ones{typeof(one(T1)*one(T2))}(d, d)
               @test @insrc(Eye{T1}(d) * Ones{T2}(d)) == Ones{typeof(one(T1)*one(T2))}(d)
               for val in (1, 2)
                   @test @insrc(Eye{T1}(d) * Fill{T2}(val, d, d)) == Fill{typeof(one(T1)*one(T2))}(val, d, d)
                   @test @insrc(Eye{T1}(d) * Fill{T2}(val, d)) == Fill{typeof(one(T1)*one(T2))}(val, d)
               end
           end
       end
    end
end

@testset "any all iszero isone" begin
    for T in (Int, Float64, ComplexF64)
        d1 = 1
        m = Eye{T}(d1)
        @test ! @inmodule FillArrays any(iszero, m)
        @test ! @ninsrc all(iszero, m)
        @test @ninsrc any(isone, m)
        @test @ninsrc all(isone, m)

        onem = Ones{T}(d1, d1)
        @test @ninsrc isone(onem)
        @test ! iszero(onem)

        zerom = Zeros{T}(d1, d1)
        @test ! isone(zerom)
        @test  iszero(zerom)

        fillm0 = Fill(T(0), d1, d1)
        @test ! isone(fillm0)
        @test   iszero(fillm0)

        fillm1 = Fill(T(1), d1, d1)
        @test isone(fillm1)
        @test ! iszero(fillm1)

        fillm2 = Fill(T(2), d1, d1)
        @test ! isone(fillm2)
        @test ! iszero(fillm2)

        for d in (2, 3)
            m = Eye{T}(d)
            @test any(iszero, m)
            @test ! all(iszero, m)
            @test any(isone, m)
            @test ! all(isone, m)

            m1 = Ones{T}(d, d)
            @test ! isone(m1)
            @test ! iszero(m1)
            @test all(isone, m1)
            @test ! all(iszero, m1)

            m2 = Zeros{T}(d, d)
            @test ! isone(m2)
            @test iszero(m2)
            @test ! all(isone, m2)
            @test  all(iszero, m2)

            m3 = Fill(T(2), d, d)
            @test ! isone(m3)
            @test ! iszero(m3)
            @test ! all(isone, m3)
            @test ! all(iszero, m3)
        end
    end
end

@testset "reduction types" begin
    for ncols in (1, 3, 10)
        for T in (Int, Float64, Int32, BigInt) # Rational{Int})
            for fop in (sum, prod)
                mi = Eye{T}(ncols)
                midense = Matrix(mi)
                r = @insrc fop(mi)
                rdense = fop(midense)
                @test r == rdense
                @test typeof(r) == eltype(mi)
            end
        end
    end
end

@testset "transform" begin
    m = Eye(3)
    for f in (triu, triu!, tril, tril!, inv)
        @test m == @insrc f(m) # FIXME: why does === fail ?
    end
end

@testset "kron Eye" begin
    for ncols1 in (1, 2, 3, 4)
        for ncols2 in (1, 2, 3, 4)
            im = Eye(ncols1)
            imd = Matrix(im)
            md = rand(ncols2, ncols2)
            @test kron(imd, md) == @insrc kron(im, md)
            @test kron(md, imd) == @insrc kron(md, im)
            @test isone(@isinsrc(kron(im, im)))
            @test isa(kron(im, im), typeof(im))
            @test size(@insrc(kron(im, zeros(0, 0)))) == (0,0)
            @test size(kron(zeros(0, 0), im)) == (0,0)
        end
    end
end

@testset "kron Diagonal" begin
    Ntests = 10
    for m in (1, 2, 3, 4)
        for m1 in (1, 2, 3, 4, 7)
            for n1 in (1, 2, 3, 4, 7)
                for icount in 1:Ntests
                    md = Diagonal(rand(m))
                    mdd = Matrix(md)
                    m2 = rand(m1, n1)
                    @test kron(md, m2) == kron(mdd, m2)
                    @test kron(m2, md) == kron(m2, mdd)
                end
            end
        end
    end
end

@testset "eigen" begin
    for ncols in (0, 1, 3, 5)
        im = Eye(ncols)
        md = Matrix(im)
        ri = @insrc eigen(im)
        rd = eigen(md)
        @test isapprox(ri.values, rd.values)
        @test isapprox(ri.vectors, rd.vectors)
        @test isapprox(eigvals(md), @insrc(eigvals(im)))
        @test isapprox(eigvecs(im), eigvecs(md))
        @test isa(ri.vectors, typeof(im))
    end
end

@testset "one oneunit zero" begin
    for ncols in (0, 1, 3, 10)
        for T in (Float64, Int)
            m = Eye{T}(ncols)
            @test one(m) === m
            @test oneunit(m) === m
            @test zero(m) == Diagonal(Zeros{eltype(m)}(size(m, 1)))
        end
    end
end

@testset "Fill" begin
    for ncols in (1, 3, 10)
        for T in (Float64,)
            for val in (1, 2, 2.5, 10)
                fm = Diagonal(Fill{T}(val, ncols))
                fmd = Matrix(fm)
                @test inv(fm) == inv(fmd)
            end
        end
    end
end

@testset "SumSpace" begin
    @testset "SumSpace bug" begin
        dsp=JacobiWeight(1.,0.,Jacobi(1.,0.,0..1))⊕JacobiWeight(0.5,0.,Jacobi(0.5,-0.5,0..1))
        rsp=Legendre(0..1)⊕JacobiWeight(0.5,0.,Jacobi(0.5,0.5,0..1))

        C=Conversion(dsp,rsp)

        f=Fun(dsp,[1.,2.,3.,4.,5.])
        @test f(0.1) ≈ (C*f)(0.1)
    end

    @testset "Triple SumSpace" begin
        x=Fun()
        w=log(1-x)+sqrt(1-x^2)
        @time f=w+x
        @test f(0.1) ≈ (w(0.1)+0.1)
        @test (w+1)(0.1) ≈ (w(0.1)+1)
        @test (w+x+1)(0.1) ≈ (w(0.1)+1.1)
        @test ((w+x)+1)(0.1) ≈ (w(0.1)+1.1)
    end

    @testset "Previoius segfault" begin
        x=Fun(identity,-1..1)
        @time f=x+sin(2x)*sqrt(1-x^2)
        @test f(0.1) ≈ 0.1+sin(2*0.1)*sqrt(1-0.1^2)
    end

    @testset "Multiple piecewisespace" begin
        x=Fun(identity,-3 .. -2)+Fun(identity,2..3)
        w=sqrt(9-x^2)
        f=w+Fun()
        @test (f+w)(2.5) ≈ 2w(2.5)
        @test (f+w)(.5) ≈ f(.5)
    end

    @testset "cancellation conversion" begin
        x=Fun(0..1)
        f=exp(x)-1
        Fun(f,JacobiWeight(1.,0.,0..1))
    end

    @testset "Union of ChebyshevDirichlet" begin
        dom = UnionDomain(0..1, 2..3)
        sps = components(union(JacobiWeight.(-0.5,-0.5,ChebyshevDirichlet{1,1}.(components(dom)))...))
        spsexpected = (JacobiWeight.(-0.5,-0.5,ChebyshevDirichlet{1,1}.(components(dom)))...,)
        @test all(((x,y),) -> x == y, zip(sps, spsexpected))
    end

    @testset "Ultraspherical special functions" begin
        x = Fun(Ultraspherical(2,0..1))
        sqrt(x)(0.1) ≈ sqrt(0.1)

        f = Fun(x->x*exp(x),Ultraspherical(1,0..1))
        sqrt(f(0.1)) ≈ sqrt(f)(0.1)
    end

    @testset "one for SumSpace" begin
        S = Jacobi(0,1) ⊕ JacobiWeight(1/3,0,Jacobi(1/3,2/3)) ⊕ JacobiWeight(2/3,0,Jacobi(2/3,1/3))
        o = ones(S)
        @test o(0.5) ≈ 1
    end

    @testset "Complex piecewise" begin
        x = Fun(identity, Segment(im,0) ∪ Segment(0,1))

        @test sqrt(1-x)(0.2im) ≈ sqrt(1-0.2im)
        @test sqrt(1-x)(0.2) ≈ sqrt(1-0.2)

        w=2/(sqrt(1-x)*sqrt(1+im*x))
        @test w(0.2im) ≈ 2/(sqrt(1-0.2im)*sqrt(1+im*(0.2im)))
        @test w(0.2) ≈ 2/(sqrt(1-0.2)*sqrt(1+im*(0.2)))
    end

    @testset "hasconversion" begin
        @test ApproxFunBase.hasconversion(Legendre() + Legendre(), JacobiWeight(0,0,Legendre()))
    end
end
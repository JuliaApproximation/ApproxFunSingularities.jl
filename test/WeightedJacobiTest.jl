@testset "WeightedJacobi" begin
    m=20
    @time testtransforms(JacobiWeight(0.,m,Jacobi(0.,2m+1)))
    f=Fun(x->((1-x)/2).^m.*exp(x),JacobiWeight(0.,m,Jacobi(0.,2m+1)))
    @test abs(f(.1)-(x->((1-x)/2).^m.*exp(x))(.1))<10eps()


    m=10
    @time f=Fun(x->besselj(m,m*(1-x)),JacobiWeight(0.,m,Jacobi(0.,2m+1)))
    @test f(0.) ≈ besselj(m,m)

    @testset "Conversion" begin
        testtransforms(Jacobi(-0.5,-0.5))
        @test norm(Fun(Fun(exp),Jacobi(-.5,-.5))-Fun(exp,Jacobi(-.5,-.5))) < 300eps()

        x=Fun(identity)
        ri=0.5/(1-x)
        @test ((1-x)/2 .* Fun(exp,JacobiWeight(0.,0.,Jacobi(0.,1.))))(.1) ≈ (1-.1)./2*exp(.1)


        @test ((1-x)/2 .* Fun(exp,JacobiWeight(0.,0.,Jacobi(0.,1.))))(.1) ≈ (1-.1)./2*exp(.1)


        @test (ri*Fun(exp,JacobiWeight(0.,0.,Jacobi(0.,1.))))(.1) ≈ .5/(1-.1)*exp(.1)
    end

    @testset "Derivative" begin
        D=Derivative(Jacobi(0.,1.,Segment(1.,0.)))
        @time testbandedoperator(D)

        S=JacobiWeight(0.,0.,Jacobi(0.,1.,Segment(1.,0.)))
        D=Derivative(S)
        testbandedoperator(D)

        f=Fun(exp,domainspace(D))
        @test (D*f-f).coefficients|>norm < eps(100000.)
        @test (f'-f).coefficients|>norm < eps(100000.)
        @test (D^2*f-f).coefficients|>norm < eps(100000000.)
        @test (D*(D*f)-f).coefficients|>norm < eps(100000000.)

        S=JacobiWeight(1,1,Ultraspherical(1))

        f=Fun(S,[1.,2.,3.])
        @test (Derivative(S,2)*f)(0.1) ≈ f''(0.1)
    end

    @testset "special derivative" begin
        x=Fun()
        f=exp(x)*sqrt(1-x^2)
        D=Derivative(WeightedJacobi(.5,.5))

        testtransforms(WeightedJacobi(.5,.5))
        testbandedoperator(D)

        @time g=(D*Fun(f,domainspace(D)))
        @test f'(0.1) ≈ g(0.1)
    end


    @testset "==" begin
        @test WeightedJacobi(0.1,0.2) == WeightedJacobi(0.1+eps(),0.2)
    end

    @testset "subspace bug" begin
        f=Fun(WeightedJacobi(0.1,0.2),rand(10))  # convert to Legendre expansion

        g=(f|(2:∞))

        @test ApproxFunBase.coefficients(g.coefficients,space(g),ApproxFunBase.canonicalspace(g))[1] ==0.
        @test norm((Fun(g,space(f))|(2:∞)-g).coefficients) < 10eps()
    end

    @testset "JacobiWeight cumsum bug Issue #557" begin
        x = Fun(0.0..1.0)
        ν = 2
        @time f = x^(ν/2-1) * exp(-x/2) # 0.05s
        @test cumsum(f)' ≈ f
        @test cumsum(f)(1.0) ≈ 0.7869386805747332 # Mathematic

        x = Fun(Ray())
        ν = 2
        @time f = x^(ν/2-1) * exp(-x/2) # 0.05s
        @test cumsum(f)' ≈ f
        @test cumsum(f)(1.0) ≈ 0.7869386805747332
    end

    @testset "Definite integral" begin
        @testset for S in (WeightedJacobi(0,0), JacobiWeight(0,0, Legendre(1.1..2.3)), Legendre())
            B = DefiniteIntegral(S)
            testfunctional(B)
            @test ApproxFunBase.rowstop(B,1) == 1
            B[1] == arclength(domain(S))
            f = Fun(exp, S)
            @test convert(Number, B*f) ≈ sum(Fun(exp,domain(S)))
        end

        f = Fun(x->√(1-x^2), JacobiWeight(0.5, 0.5, NormalizedJacobi(1,1)))
        g = Fun(x->√(1-x^2), JacobiWeight(0.5, 0.5, Jacobi(1,1)))
        @test sum(f) ≈ sum(g) ≈ π/2
        @test DefiniteIntegral() * f ≈ DefiniteIntegral() * g ≈ π/2
    end
end
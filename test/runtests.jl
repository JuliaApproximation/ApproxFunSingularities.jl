module AFSTests

using ApproxFunBase
using ApproxFunBase: HeavisideSpace, PointSpace, ArraySpace, DiracSpace, PiecewiseSegment,
                        UnionDomain, resizedata!, CachedOperator, RaggedMatrix,
                        Block, ∞, BandedBlockBandedMatrix, NoSpace, ConcreteMultiplication,
                        MultiplicationWrapper
using ApproxFunBaseTest: testbandedoperator, testtransforms, testfunctional,
                        testbandedblockbandedoperator
using ApproxFunOrthogonalPolynomials
using ApproxFunOrthogonalPolynomials: order
using ApproxFunSingularities
using HalfIntegers
using IntervalSets
using LinearAlgebra
using OddEvenIntegers
using SpecialFunctions
using Test

using Aqua
@testset "Project quality" begin
    Aqua.test_all(ApproxFunSingularities, ambiguities=false,
        stale_deps=(; ignore=[:ApproxFunBaseTest]), piracy = false,
        # only test formatting on VERSION >= v1.7
        # https://github.com/JuliaTesting/Aqua.jl/issues/105#issuecomment-1551405866
        project_toml_formatting = VERSION >= v"1.9")
end

@testset "utility function" begin
    @test ApproxFunSingularities.isapproxhalfoddinteger(0.5)
    @test ApproxFunSingularities.isapproxhalfoddinteger(half(1))
    @test ApproxFunSingularities.isapproxhalfoddinteger(half(Odd(1)))
    @test !ApproxFunSingularities.isapproxhalfoddinteger(1)
end

@testset "Sqrt" begin
    x=Fun(identity);
    @time @test sqrt(cos(π/2*x))(.1) ≈ sqrt(cos(.1π/2))

    x=Fun(identity,-2..2)
    @time u=sqrt(4-x^2)/(2π)

    @test u(.1) ≈ sqrt(4-0.1^2)/(2π)
    @test sum(u) ≈ 1

    #this call threw an error, which we check
    @test length(values(u)) == 1

    f = Fun(x->x*cot(π*x/2))
    x = Fun(identity)
    u = Fun(JacobiWeight(1.,1.,ChebyshevInterval()), (f/(1-x^2)).coefficients)
    @test 1/(0.1*cot(π*0.1/2)) ≈ (1/u)(.1)

    @test (x/u)(.1) ≈ tan(π*.1/2)

    f=Fun(x->exp(-x^2),Line(0.,0.,-.5,-.5),400)
    @time @test sum(f) ≈ sqrt(π)

    f=Fun(x->exp(x)/sqrt(1-x.^2),JacobiWeight(-.5,-.5))
    @test f(.1) ≈ (x->exp(x)/sqrt(1-x.^2))(.1)

    @time @test norm(Fun(exp,Legendre(0..1))+sqrt(Fun(0..1))) ≈ 2.491141949903508

    @testset "sampling Chebyshev" begin
        x=Fun(identity)
        f = exp(x)/sqrt(1-x^2)
        @time g = cumsum(f)
        @test abs(g(-1)) ≤ 1E-15
        @test g'(0.1) ≈ f(0.1)
    end

    @testset "Complex domains sqrt" begin
        a=1+10*im; b=2-6*im
        d = IntervalCurve(Fun(x->1+a*x+b*x^2))

        x=Fun(d)
        w=sqrt(abs(leftendpoint(d)-x))*sqrt(abs(rightendpoint(d)-x))

        @test sum(w/(x-2.))/(2π*im) ≈ (-4.722196879007759+2.347910413861846im)
        @test linesum(w*log(abs(x-2.)))/π ≈ (88.5579588360686)

        a=Arc(0.,1.,0.,π/2)
        ζ=Fun(identity,a)
        f=Fun(exp,a)*sqrt(abs((ζ-1)*(ζ-im)))
    end

    @time @test norm(Fun(exp,Legendre(0..1))+sqrt(Fun(0..1))) ≈ 2.491141949903508
end

@testset "JacobiWeight" begin
    @testset "Sub-operator re-view bug" begin
        D = Derivative(Chebyshev())
        S = view(D[:, 2:end], Block.(3:4), Block.(2:4))
        @test parent(S) == D
        @test parentindices(S) == (3:4,2:4)
        @test bandwidths(S)  == (-2,2)

        DS=JacobiWeight(1,1,Jacobi(1,1))
        D=Derivative(DS)[2:end,:]
        @test domainspace(D) == DS | (1:∞)
        testbandedoperator(D)
    end

    @testset "Multiplication functions" begin
        x = Fun()
        M = Multiplication(x, JacobiWeight(0,0,Chebyshev()))
        @test exp(M).f == Multiplication(exp(x), Chebyshev()).f

        g = Fun(x->√(1-x^2), JacobiWeight(0.5, 0.5, Jacobi(1,1)))
        xg = Fun(x->x*√(1-x^2), JacobiWeight(0.5, 0.5, Jacobi(1,1)))
        @test Multiplication(g) * Fun(NormalizedLegendre()) ≈ xg

        genf(β,α) = Fun(x -> (1+x)^β * (1-x)^α, JacobiWeight(β,α,ConstantSpace(ChebyshevInterval())));

        f = genf(0,2)
        S = Jacobi(5,5)
        d = domain(S)
        # Multiplication(f, S) is inferred as a small Union
        # We enumerate the possible types
        T1 = typeof(Multiplication(genf(1,0), S)::ConcreteMultiplication)
        T2 = typeof(Multiplication(f, S)::MultiplicationWrapper)
        T3 = typeof(Multiplication(genf(1,0), Legendre())::MultiplicationWrapper)
        @inferred Union{T1,T2,T3} Multiplication(f, S)

        dsp(f, S) = domainspace(Multiplication(f, S))
        rsp(f, S) = rangespace(Multiplication(f, S))
        ds = if VERSION >= v"1.9"
            @inferred dsp(f, S)
        else
            dsp(f, S)
        end
        @test ds == S
        @test rsp(f, S) == Jacobi(5,3)

        f = genf(4,3)
        S = Jacobi(2,1)
        @test rsp(f, S) == JacobiWeight(2,2,Legendre())

        f = genf(half(Odd(3)), half(Odd(3)))
        S = Jacobi(2,0)
        @test @inferred(((f,S) -> domainspace(@inferred Multiplication(f, S)))(f,S)) == S
        @test Multiplication(f, S) * Fun(S) ≈ Fun(x->x*(1-x^2)^(3/2), JacobiWeight(3/2, 3/2, S))

        S = Jacobi(half(Odd(1)), half(Odd(3)))
        @test @inferred(domainspace(@inferred Multiplication(f,S))) == S

        @testset for β in 0:0.5:5, α in 0:0.5:5
            f = genf(β, α)
            g = Fun(x->(1+x)^2 * (1+x)^β * (1-x)^α, JacobiWeight(β+2, α, Chebyshev()))
            @testset for b in 0:8, a in 0:8
                S = Jacobi(b,a)
                w = Fun(x->(1+x)^2, S)
                M = Multiplication(f, S)
                @test domainspace(M) == S
                if isinteger(β) && isinteger(α) && b >= β && a >= α
                    @test rangespace(M) == Jacobi(b-β, a-α)
                elseif isinteger(β) && isinteger(α) && b < β && a < α
                    @test rangespace(M) == JacobiWeight(β-b, α-a, Legendre())
                elseif !isinteger(β) && (!isinteger(α) || α == 0)
                    @test rangespace(M) == JacobiWeight(β, α, S)
                elseif !isinteger(α) && (!isinteger(β) || β == 0)
                    @test rangespace(M) == JacobiWeight(β, α, S)
                end
                @test Multiplication(f) * w ≈ M * w ≈ g
            end
        end
    end

    @testset "Derivative" begin
        S = JacobiWeight(-1.,-1.,Chebyshev(0..1))

        # Checks bug in Derivative(S)
        @test typeof(ConstantSpace(0..1)) <: Space{ClosedInterval{Int},Float64}

        D=Derivative(S)
        f=Fun(S,Fun(exp,0..1).coefficients)
        x=0.1
        @test f(x) ≈ exp(x)*x^(-1)*(1-x)^(-1)/4
        @test (D*f)(x) ≈ -exp(x)*(1+(x-3)*x)/(4*(x-1)^2*x^2)


        S=JacobiWeight(-1.,0.,Chebyshev(0..1))
        D=Derivative(S)

        f=Fun(S,Fun(exp,0..1).coefficients)
        x=.1
        @test f(x) ≈ exp(x)*x^(-1)/2
        @test (D*f)(x) ≈ exp(x)*(x-1)/(2x^2)
    end

    @testset "differentiate" begin
        f = Fun(x -> √(1-x^2) * x^2, JacobiWeight(0.5, 0.5, Chebyshev()))
        df(f) = ApproxFunSingularities.differentiate(f)
        g = if VERSION >= v"1.9"
                @inferred df(f)
            else
                df(f)
            end

        @test g ≈ Fun(x -> -x^3/√(1-x^2) + √(1-x^2) * 2x, JacobiWeight(-0.5, -0.5, Chebyshev()))
    end

    @testset "Jacobi singularity" begin
        x = Fun(identity)
        f = exp(x)/(1-x.^2)

        @test f(.1) ≈ exp(.1)/(1-.1^2)
        f = exp(x)/(1-x.^2).^1
        @test f(.1) ≈ exp(.1)/(1-.1^2)
        f = exp(x)/(1-x.^2).^1.0
        @test f(.1) ≈ exp(.1)/(1-.1^2)

        ## 1/f with poles
        x=Fun(identity)
        f=sin(10x)
        g=1/f

        @test g(.123) ≈ csc(10*.123)
    end

    @testset "Jacobi conversions" begin
        S1,S2=JacobiWeight(3.,1.,Jacobi(1.,1.)),JacobiWeight(1.,1.,Jacobi(0.,1.))
        f=Fun(S1,[1,2,3.])
        C=Conversion(S1,S2)
        Cf=C*f
        @test Cf(0.1) ≈ f(0.1)

        S1,S2=JacobiWeight(3.,2.,Jacobi(1.,1.)),JacobiWeight(1.,1.,Jacobi(0.,0.))
        f=Fun(S1,[1,2,3.])
        C=Conversion(S1,S2)
        Cf=C*f
        @test Cf(0.1) ≈ f(0.1)
    end

    @testset "Array Conversion" begin
        a = ArraySpace(JacobiWeight(1/2,1/2, Chebyshev()), 2)
        b = ArraySpace(JacobiWeight(1/2,1/2, Ultraspherical(1)), 2)
        C = Conversion(a, b)

        f = Fun(a, rand(10))
        @test f(0.1) ≈ (C*f)(0.1)

        a = ArraySpace(JacobiWeight(1/2,1/2, Chebyshev()), 2,3)
        b = ArraySpace(JacobiWeight(1/2,1/2, Ultraspherical(1)), 2,3)
        C = Conversion(a, b)

        f = Fun(a, rand(10))
        @test f(0.1) ≈ (C*f)(0.1)
    end

    @testset "Equivalent spaces" begin
        @test norm(Fun(cos,Chebyshev)-Fun(cos,Jacobi(-0.5,-0.5)))<100eps()
        @test norm(Fun(cos,Chebyshev)-Fun(cos,JacobiWeight(0,0)))<100eps()
        @test norm(Fun(cos,Jacobi(-0.5,-0.5))-Fun(cos,JacobiWeight(0,0))) < 100eps()
        @test norm(Fun(cos,Chebyshev)-Fun(cos,JacobiWeight(0,0,Jacobi(-0.5,-0.5))))<100eps()
        @test norm(Fun(cos,Jacobi(-0.5,-0.5))-Fun(cos,JacobiWeight(0,0,Jacobi(-0.5,-0.5))))<100eps()
    end

    @testset "Ultraspherical order" begin
        us = Ultraspherical(0.5)
        s = JacobiWeight(1, 1, us)
        @test order(s) == order(us)
    end

    @testset "PiecewiseSpace" begin
        ps = PiecewiseSpace((Chebyshev(0..1),Chebyshev(1..2)))
        jps = JacobiWeight(0,0,ps)
        @test domain(jps) == domain(ps)
    end

    @testset "inference in maxspace" begin
        sp = JacobiWeight(half(Odd(1)), half(Odd(1)), Legendre())
        @test (@inferred maxspace(sp, sp)) == sp

        sp2 = JacobiWeight(half(Odd(1)), half(Odd(1)), Legendre(0..1))
        @test maxspace(sp, sp2) == NoSpace()

        @test (@inferred maxspace(sp, Legendre())) == NoSpace()

        sp = JacobiWeight(0.5,1,Legendre())
        @test (@inferred Union{typeof(sp),NoSpace} ApproxFunBase.maxspace_rule(sp, sp)) == sp
    end

    @testset "Evaluation bug" begin
        S = Chebyshev()
        E = Evaluation(S, 0.5)
        EJW = Evaluation(JacobiWeight(0,0,S), 0.5)
        @test EJW[4] ≈ E[4]
        @test EJW[1:4] ≈ E[1:4]
    end
end

@testset "Ray and Line" begin
    @testset "broadcast!" begin
        f = Fun()
        @test_throws ArgumentError (f .= Fun(Line()))
    end
    @testset "Ray" begin
        @test Inf in Ray()   # this was a bug
        @test Space(0..Inf) == Chebyshev(Ray())
        f=Fun(x->exp(-x),0..Inf)
        @test f(0.1) ≈ exp(-0.1)
        @test f'(.1) ≈ -f(.1)

        x=Fun(identity,Ray())
        f=exp(-x)
        u=integrate(f)
        @test (u(1.)-u(0)-1) ≈ -f(1)

        x=Fun(identity,Ray())
        f=x^(-0.123)*exp(-x)
        @test integrate(f)'(1.) ≈ f(1.)

        @test ≈(sum(Fun(sech,0..Inf)),sum(Fun(sech,0..40));atol=1000000eps())
        @test Line() ∪ Ray() == Line()
        @test Line() ∩ Ray() == Ray()

        f=Fun(sech,Line())
        @test Fun(f,Ray())(2.0) ≈ sech(2.0)
        @test Fun(f,Ray(0.,π))(-2.0) ≈ sech(-2.0)
        @test Fun(sech,Ray(0.,π))(-2.0) ≈ sech(-2.0)
    end

    @testset "Ei (Exp Integral)" begin
        y=Fun(Ray())
        q=integrate(exp(-y)/y)
        @test (q-last(q))(2.) ≈ (-0.04890051070806113)

        ## Line
        f=Fun(x->exp(-x^2),Line())

        @test f'(0.1) ≈ -2*0.1exp(-0.1^2)
        @test (Derivative()*f)(0.1) ≈ -2*0.1exp(-0.1^2)
    end
end

@testset "LogWeight" begin
    x=Fun(identity,-1..1)
    f=exp(x+1)-1
    @test log(f)(0.1) ≈ log(f(0.1))

    x=Fun(identity,0..1)
    f=exp(x)-1
    @test log(f)(0.1) ≈ log(f(0.1))

    x=Fun(identity,0..1)
    @test Fun(exp(x)/x-1/x,Chebyshev)(0.1) ≈ (exp(0.1)-1)/0.1

    x=Fun(identity,0..1)
    f=1/x
    p=integrate(f)
    @test (p-p(1.))(0.5) ≈ log(0.5)

    f=1/(1-x)
    p=integrate(f)
    @test (p-p(0.))(0.5) ≈ -log(1-0.5)

    @testset "#393" begin
        x = Fun(0..1)
        @time f = exp(x)*sqrt(x)*log(1-x)
        @test f(0.1) ≈ exp(0.1)*sqrt(0.1)*log(1-0.1)
    end
end

@testset "ExpWeight" begin
    S = ExpWeight(-Fun()^2, Chebyshev())
    x = 0.5
    n = 2
    @test S(n, x) ≈ exp(-x^2) * Chebyshev()(n, x)
    @test S(n)'(x) ≈ exp(-x^2) * Chebyshev()(n)'(x) - 2x * exp(-x^2) * Chebyshev()(n, x)
end

@testset "DiracDelta and PointSpace" begin
    a,b=DiracDelta(0.),DiracDelta(1.)
    f=Fun(exp)
    g=a+0.2b+f
    @test components(g)[2](0.) ≈ 1.
    @test g(.1) ≈ exp(.1)
    @test sum(g) ≈ (sum(f)+1.2)

    #Checks prevoius bug
    δ=DiracDelta()
    x=Fun()
    w=sqrt(1-x^2)
    @test (w+δ)(0.1) ≈ w(0.1)
    @test sum(w+δ) ≈ sum(w)+1

    ## PointSpace
    f=Fun(x->(x-0.1),PointSpace([0,0.1,1]))
    g = f + Fun(2..3)
    @test f(0.0) ≈ g(0.0) ≈ -0.1
    @test f(0.1) ≈ g(0.1) ≈ 0.0
    @test f(1.0) ≈ g(1.0) ≈ 0.9

    @test g(2.3) ≈ 2.3

    h = a + Fun(2..3)

    # for some reason this test is broken only on Travis
    @test_skip g/h ≈ f/a + Fun(1,2..3)
end

@testset "Multiple roots" begin
    x=Fun(identity,-1..1)

    @test (1/x^2)(0.1) ≈ 100.
    @test (1/x^2)(-0.1) ≈ 100.

    fc=x*(1+x)^2
    @time @test (1/fc)(0.1) ≈ 1/fc(0.1)

    fc=x*(1-x)^2
    @test (1/fc)(0.1) ≈ 1/fc(0.1)
end

@testset "special function singularities" begin
    x=Fun(0..1)
    @time @test erf(sqrt(x))(0.1) ≈ erf(sqrt(0.1))
    @time @test erfc(sqrt(x))(0.1) ≈ erfc(sqrt(0.1))

    ## roots of log(abs(x-y))
    x=Fun(-2..(-1))
    @test space(abs(x)) == Chebyshev(-2 .. (-1))

    @test roots(abs(x+1.2)) ≈ [-1.2]

    f = sign(x+1.2)
    @test space(f) isa PiecewiseSpace
    @test f(-1.4) == -1
    @test f(-1.1) == 1

    f=abs(x+1.2)
    @test abs(f)(-1.3) ≈ f(-1.3)
    @test abs(f)(-1.1) ≈ f(-1.1)
    @test norm(abs(f)-f)<10eps()

    @test norm(sign(f)-Fun(1,space(f)))<10eps()

    @test log(f)(-1.3) ≈ log(abs(-1.3+1.2))
    @test log(f)(-1.1) ≈ log(abs(-1.1+1.2))
end

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
        @time f = x^(ν/2-1) * exp(-x/2) # 0.05s
        @test cumsum(f)' ≈ f
        @test cumsum(f)(1.0) ≈ 0.7869386805747332 # Mathematic

        x = Fun(Ray())
        ν = 2
        @time f = x^(ν/2-1) * exp(-x/2) # 0.05s
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

@testset "WeightedLaguerre" begin
    @testset "WeightedLaguerre cumsum" begin
        α = 2.7
        f = Fun(WeightedLaguerre(α), [1.0]);
        f = Fun(f, JacobiWeight(α,0,Chebyshev(Ray())));
        g = integrate(f)
        g(3.0) - cumsum(Fun(x -> f(x), 0..6))(3.0)
    end

    @testset "Log with squareroot singularities" begin
        a = 1.0; b = 2.0+im
        d = Segment(a,b)
        z = Fun(d)

        f = real(exp(z) / (sqrt(z-a)*sqrt(b-z)))
        S=space(f)
        x=4.0+2im;
        @test linesum(f*log(abs(x-z))) ≈ 13.740676344264614
    end
end

@testset "Hermite Integration" begin
    @test_throws ArgumentError integrate(Fun(GaussWeight(Hermite(2),1), [0.0,1.0]))

    w = Fun(GaussWeight(Hermite(2), 0), [1.0,2.0,3.0])
    g = integrate(w)
    g̃ = Fun(Hermite(2), [0.0, 0.5, 0.5, 0.5])
    @test g(0.1) == g̃(0.1)

    w = Fun(GaussWeight(), Float64[])
    g = integrate(w)
    @test g(0.1) == 0.0

    w = Fun(GaussWeight(), [1.0])
    g = integrate(w)
    @test_skip w̃ = Fun(w, -7..7)
    w̃ = Fun( x-> w(x), -7..7)
    g̃ = cumsum(w̃)
    @test g(3) - g(-7) ≈ g̃(3)

    w = Fun(GaussWeight(), Float64[1.0])
    g = integrate(w)
    @test_skip w̃ = Fun(w, -7..7)
    w̃ = Fun(x -> w(x), -7..7)
    g̃ = cumsum(w̃)
    @test g(3) - g(-7) ≈ g̃(3)

    w = Fun(GaussWeight(Hermite(2), 2), Float64[1.0])
    g = integrate(w)
    @test_skip w̃ = Fun(w, -7..7)
    w̃ = Fun(x -> w(x), -7..7)
    g̃ = cumsum(w̃)
    @test g(3) - g(-7) ≈ g̃(3)

    w = Fun(GaussWeight(), Float64[0.0, 1.0])
    g = integrate(w)
    @test_skip w̃ = Fun(w, -7..7)
    w̃ = Fun(x -> w(x), -7..7)
    g̃ = cumsum(w̃)
    @test g(3) - g(-7) ≈ g̃(3)

    w = Fun(GaussWeight(Hermite(2), 2), Float64[0.0, 1.0])
    g = integrate(w)
    @test_skip w̃ = Fun(w, -7..7)
    w̃ = Fun(x -> w(x), -7..7)
    g̃ = cumsum(w̃)
    @test g(3) - g(-7) ≈ g̃(3)
end

@testset "Laguerre" begin
    @testset "Integration" begin
        @test_throws ArgumentError integrate(Fun(LaguerreWeight(1, Laguerre(2)), [1.0, 2.0]))

        g = integrate(Fun(WeightedLaguerre(), []))
        @test g(0.1) == 0.0

        α = 2.8
        f = Fun(WeightedLaguerre(α), [1.0])
        g = integrate(f)
        @test g(3.0) - cumsum(Fun(x -> f(x), 0..6))(3.0) ≈ g(0.0)

        α = 2
        f = Fun(WeightedLaguerre(α), [1.0])
        g = integrate(f)
        @test g(3.0) - cumsum(Fun(x -> f(x), 0..6))(3.0) ≈ g(0.0)

        α = 2.8
        f = Fun(WeightedLaguerre(α), [0.0, 1.0])
        g = integrate(f)
        f̃ = Fun(x -> f(x), 0 .. 100)
        g̃ = integrate(f̃)
        g̃ = g̃ - last(g̃)
        @test g(3.0) ≈ g̃(3.0)

        α = 2
        f = Fun(WeightedLaguerre(α), [1.0, 1.0])
        g = integrate(f)
        @test g(3.0) - cumsum(Fun(x -> f(x), 0..6))(3.0) ≈ g(0.0)

        α = 5
        f = Fun(WeightedLaguerre(α), [1.0])
        g = integrate(f)
        @test g(3.0) - cumsum(Fun(x -> f(x), 0..6))(3.0) ≈ g(0.0)

        α = 5
        f = Fun(WeightedLaguerre(α), [0.0, 1.0])
        g = integrate(f)
        f̃ = Fun(x -> f(x), 0 .. 100)
        g̃ = integrate(f̃)
        g̃ = g̃ - last(g̃)
        @test g(3.0) ≈ g̃(3.0)

        α = 5
        f = Fun(WeightedLaguerre(α), [1.0, 1.0])
        g = integrate(f)
        @test g(3.0) - cumsum(Fun(x -> f(x), 0..6))(3.0) ≈ g(0.0)
    end

    @testset "Correct domain" begin
        w = Fun(WeightedLaguerre(0.5),[1.0])
        h = cumsum(w)
        @test domain(h) == Ray()
    end
end


include("IntegralEquationsTest.jl")

@testset "PDE" begin
    @testset "Laplacian" begin
        S=WeightedJacobi(1,1)^2
        L=Laplacian(S)
        testbandedblockbandedoperator(L)
    end
    @testset "Zero Dirichlet" begin
        S = JacobiWeight(1.,1.,Jacobi(1.,1.))^2
        Δ = Laplacian(S)

        testbandedblockbandedoperator(Δ)

        u = Fun((x,y)->sin(π*x)*sin(π*y),S)
        f = -2π^2*u

        F = qr(Δ)
        ApproxFunBase.resizedata!(F,:,1000)
        @time v=F\f
        @test norm((u-v).coefficients)<100eps()


        F=qr(Δ)
        ApproxFunBase.resizedata!(F.R_cache,:,100)
        ApproxFunBase.resizedata!(F.R_cache,:,1000)
        @time v=F \ f
        @test norm((u-v).coefficients)<100eps()

        F=qr(Δ)
        @time v=F\f
        @test norm((u-v).coefficients)<100eps()
    end
    @testset "Gaussian zero Dirichlet Poisson" begin
        S=JacobiWeight(1.,1.,Jacobi(1.,1.))^2
        Δ=Laplacian(S)

        f=Fun((x,y)->exp(-10(x+.2)^2-20(y-.1)^2),rangespace(Δ))  #default is [-1,1]^2
        @time v = \(Δ,f;tolerance=1E-14)
        @test norm((Δ*v-f).coefficients) < 1E-14

    end
    @testset "check we dispatch correctly to get fast build" begin
        S = JacobiWeight(1.,1.,Jacobi(1.,1.))^2
        Δ = Laplacian(S)
        @time S = view(Δ.op.ops[1].ops[1].op,Block.(1:40), Block.(1:40))
        @test typeof(S.parent.domaintensorizer) == ApproxFunBase.Trivial2DTensorizer
    end

    @testset "Operator resize" begin
        S=ChebyshevDirichlet()^2
        B=Dirichlet(S)
        f = Fun((x,y)->exp(x)*sin(y),S)
        @test norm((Fun((x,y)->exp(x)*sin(y),∂(domain(S))) - B*f).coefficients) < 100eps()


        S=JacobiWeight(1.,1.,Jacobi(1.,1.))^2
        Δ=Laplacian(S)

        @test cache(Δ)[1:100,1:100]  ≈ Δ[1:100,1:100]
        @test cache(Δ;padding=true)[1:100,1:100]  ≈ Δ[1:100,1:100]

        @test cache(Δ)[5:100,7:100]  ≈ Δ[5:100,7:100]
        @test cache(Δ;padding=true)[5:100,7:100]  ≈ Δ[5:100,7:100]

        # Check that QR is growing correctly
        for col in (1,2,3,10,11,40)
            QR=qr(Δ)
            resizedata!(QR.R_cache,:,col+100)
            resizedata!(QR,:,col)
            QR2=qr!(CachedOperator(RaggedMatrix,Δ;padding=true))
            resizedata!(QR2.R_cache,:,QR.ncols+100)
            resizedata!(QR2,:,QR.ncols)
            n=min(size(QR.H,1),size(QR2.H,1))
            @test QR.H[1:n,1:col] ≈ QR2.H[1:n,1:col]
            @test QR.R_cache[1:col,1:col] ≈ QR2.R_cache[1:col,1:col]
            @test QR.R_cache[1:col+10,1:col+10] ≈ QR2.R_cache[1:col+10,1:col+10]
        end

        QRR=qr(Δ)
        QR2=qr!(CachedOperator(RaggedMatrix,Δ;padding=true))
        for col in (80,200)
            resizedata!(QRR,:,col)
            resizedata!(QR2,:,QRR.ncols)
            n=min(size(QRR.H,1),size(QR2.H,1))
            @test QRR.H[1:n,1:col] ≈ QR2.H[1:n,1:col]
            @test QRR.R_cache[1:col,1:col] ≈ QR2.R_cache[1:col,1:col]
            @test QRR.R_cache[1:col+10,1:col+10] ≈ QR2.R_cache[1:col+10,1:col+10]
        end

        # this checks a bug
        QRR=qr(Δ)
        resizedata!(QRR,:,548)
        resizedata!(QRR,:,430)


        u=Fun((x,y)->sin(π*x)*sin(π*y),S)
        f=-2π^2*u


        QRR=qr(Δ)
        v=QRR\f
        @test norm((u-v).coefficients)<100eps()

        v=Δ\f
        @test norm((u-v).coefficients)<100eps()


        f=Fun((x,y)->exp(-10(x+.2)^2-20(y-.1)^2),rangespace(Δ))  #default is [-1,1]^2
        @time v=\(Δ,f;tolerance=1E-14)
        @test norm((Δ*v-f).coefficients)<1E-14

        KO=Δ.op.ops[1].ops[1].op

        M=BandedBlockBandedMatrix(view(KO,1:4,1:4))
        @test norm(BandedBlockBandedMatrix(view(KO,1:4,2:4))-M[:,2:4]) < 10eps()
        @test norm(BandedBlockBandedMatrix(view(KO,1:4,3:4))-M[:,3:4]) < 10eps()
    end
end


@testset "Cauchy fun" begin
    f = Fun((x,y)->1/(2π*(x^2+y^2+1)^(3/2)),Line()^2)
    @test f(0.1,0.2) ≈ 1/(2π*(0.1^2+0.2^2+1)^(3/2))

    #TODO: improve tolerance
    f = LowRankFun((x,y)->1/(2π*(x^2+y^2+1)^(3/2)),JacobiWeight(2.,2.,Line())^2)
    @test ≈(f(0.1,0.2),1/(2π*(0.1^2+0.2^2+1)^(3/2));atol=1E-4)
end

@testset "Bessel" begin
    @time for ν in (1.,0.5,2.,3.5)
        println("        ν = $ν")
        S=JacobiWeight(-ν,0.,Chebyshev(0..1))
        D=Derivative(S)
        x=Fun(identity,domain(S))
        L=(x^2)*D^2+x*D+(x^2-ν^2);
        u=\([rdirichlet(S);rneumann(S);L],[bessely(ν,1.),.5*(bessely(ν-1.,1.)-bessely(ν+1.,1.)),0];
                    tolerance=1E-10)
        @test ≈(u(.1),bessely(ν,.1);atol=eps(1000000.)*max(abs(u(.1)),1))
        u=Fun(x->bessely(ν,x),S)
        @test ≈(u(.1),bessely(ν,.1);atol=eps(10000.)*max(abs(u(.1)),1))
        u=Fun(x->besselj(ν,x),S)
        @test ≈(u(.1),besselj(ν,.1);atol=eps(10000.)*max(abs(u(.1)),1))
    end

    @time for ν in (1.,0.5,0.123,3.5)
        println("        ν = $ν")
        S=JacobiWeight(ν,0.,Chebyshev(0..1))
        D=Derivative(S)
        x=Fun(identity,domain(S))
        L=(x^2)*D^2+x*D+(x^2-ν^2);

        u=\([rdirichlet(S);rneumann(S);L],[besselj(ν,1.),.5*(besselj(ν-1.,1.)-besselj(ν+1.,1.)),0];
                    tolerance=1E-10)
        @test ≈(u(.1),besselj(ν,.1);atol=eps(1000000.)*max(abs(u(.1)),1))
        u=Fun(x->besselj(ν,x),S)
        @test ≈(u(.1),besselj(ν,.1);atol=eps(10000.)*max(abs(u(.1)),1))
    end
end

@testset "Speed test" begin
    S = JacobiWeight(1.,1.,Jacobi(1.,1.))^2
    Δ = Laplacian(S)

    f = Fun((x,y)->sin(π*x)*sin(π*y),S)

    QR1=qr(Δ)
    ApproxFunBase.resizedata!(QR1,:,400)
        \(QR1,f; tolerance=1E-10)
    QR1=qr(Δ)
        @time Δ[Block.(1:40), Block.(1:40)]
        @time ApproxFunBase.resizedata!(QR1,:,400)
        @time \(QR1,f; tolerance=1E-10)
    println("Laplace Dirichlet: should be ~0.015, 0.015, 0.001")
end

@testset "show" begin
    @testset "logweight" begin
        w = LogWeight(1.0, 0.0, Chebyshev())
        s = repr(w)
        @test startswith(s, "log")
        @test contains(s, "Chebyshev()")
        @test contains(s, "(1+x)^1")
        @test contains(s, "(1-x)^0")

        w = LogWeight(1.0, 1.0, Chebyshev())
        s = repr(w)
        @test startswith(s, "log")
        @test contains(s, "Chebyshev()")
        @test contains(s, "(1-x^2)^1")

        w = LogWeight(1, 2, Chebyshev(0..1))
        s = repr(w)
        @test startswith(s, "log")
        @test contains(s, "Chebyshev($(0..1))")
        @test contains(s, "(1+𝑪($(0..1), x))^1")
        @test contains(s, "(1-𝑪($(0..1), x))^2")
    end
    @testset "JacobiWeight" begin
        w = JacobiWeight(1.0, 0.0, Chebyshev())
        s = repr(w)
        @test contains(s, "Chebyshev()")
        @test contains(s, "(1+x)^1")

        w = JacobiWeight(0.0, 1.0, Chebyshev())
        s = repr(w)
        @test contains(s, "Chebyshev()")
        @test contains(s, "(1-x)^1")

        w = JacobiWeight(1.0, 1.0, Chebyshev())
        s = repr(w)
        @test contains(s, "Chebyshev()")
        @test contains(s, "(1-x^2)^1")

        w = JacobiWeight(1.0, 2.0, Chebyshev())
        s = repr(w)
        @test contains(s, "Chebyshev()")
        @test contains(s, "(1+x)^1")
        @test contains(s, "(1-x)^2")

        w = JacobiWeight(1,2,Chebyshev(0..1))
        s = repr(w)
        @test contains(s, "Chebyshev($(0..1))")
        @test contains(s, "(1+𝑪($(0..1), x))^1")
        @test contains(s, "(1-𝑪($(0..1), x))^2")
    end
end

end # module

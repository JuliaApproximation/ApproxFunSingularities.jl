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

        @testset for β in -3:0.5:5, α in -3:0.5:5
            f = genf(β, α)
            g = Fun(x->(1+x)^2 * (1+x)^β * (1-x)^α, JacobiWeight(β+2, α, Chebyshev()))
            @testset for b in 0:8, a in 0:8
                S = Jacobi(b,a)
                w = Fun(x->(1+x)^2, S)
                M = Multiplication(f, S)
                @test domainspace(M) == S
                reduceorders = (β ≥ 1 && b > 0) || (α ≥ 1 && a >0)
                if isinteger(β) && isinteger(α) && reduceorders && b >= β >= 0 && a >= α >= 0
                    @test rangespace(M) == Jacobi(b-β, a-α)
                elseif isinteger(β) && isinteger(α) && reduceorders && 0 <= b < β && 0 <= a < α
                    @test rangespace(M) == JacobiWeight(β-b, α-a, Legendre())
                elseif !isinteger(β) && !(isinteger(α) && α >= 1) ||
                        !isinteger(α) && !(isinteger(β) && β >= 1)
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

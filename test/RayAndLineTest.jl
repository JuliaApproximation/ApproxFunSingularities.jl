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

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
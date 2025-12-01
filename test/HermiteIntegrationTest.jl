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

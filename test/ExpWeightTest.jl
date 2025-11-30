@testset "ExpWeight" begin
    S = ExpWeight(-Fun()^2, Chebyshev())
    x = 0.5
    n = 2
    @test S(n, x) ≈ exp(-x^2) * Chebyshev()(n, x)
    @test S(n)'(x) ≈ exp(-x^2) * Chebyshev()(n)'(x) - 2x * exp(-x^2) * Chebyshev()(n, x)
end

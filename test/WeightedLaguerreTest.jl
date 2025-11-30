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
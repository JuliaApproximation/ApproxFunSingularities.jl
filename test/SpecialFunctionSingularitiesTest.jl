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

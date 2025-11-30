@testset "Multiple roots" begin
    x=Fun(identity,-1..1)

    @test (1/x^2)(0.1) ≈ 100.
    @test (1/x^2)(-0.1) ≈ 100.

    fc=x*(1+x)^2
    @time @test (1/fc)(0.1) ≈ 1/fc(0.1)

    fc=x*(1-x)^2
    @test (1/fc)(0.1) ≈ 1/fc(0.1)
end

@testset "utility function" begin
    @test ApproxFunSingularities.isapproxhalfoddinteger(0.5)
    @test ApproxFunSingularities.isapproxhalfoddinteger(half(1))
    @test ApproxFunSingularities.isapproxhalfoddinteger(half(Odd(1)))
    @test !ApproxFunSingularities.isapproxhalfoddinteger(1)
end

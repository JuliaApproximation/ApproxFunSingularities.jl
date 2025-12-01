@testset "Cauchy fun" begin
    f = Fun((x,y)->1/(2π*(x^2+y^2+1)^(3/2)),Line()^2)
    @test f(0.1,0.2) ≈ 1/(2π*(0.1^2+0.2^2+1)^(3/2))

    #TODO: improve tolerance
    f = LowRankFun((x,y)->1/(2π*(x^2+y^2+1)^(3/2)),JacobiWeight(2.,2.,Line())^2)
    @test ≈(f(0.1,0.2),1/(2π*(0.1^2+0.2^2+1)^(3/2));atol=1E-4)
end

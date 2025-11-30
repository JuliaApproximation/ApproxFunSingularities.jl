@testset "show" begin
    @testset "logweight" begin
        w = LogWeight(1.0, 0.0, Chebyshev())
        s = repr(w)
        @test startswith(s, "log")
        @test contains(s, "Chebyshev()")
        @test contains(s, "(1+x)^1")
        @test contains(s, "(1-x)^0")

        w = LogWeight(1.0, 1.0, Chebyshev())
        s = repr(w)
        @test startswith(s, "log")
        @test contains(s, "Chebyshev()")
        @test contains(s, "(1-x^2)^1")

        w = LogWeight(1, 2, Chebyshev(0..1))
        s = repr(w)
        @test startswith(s, "log")
        @test contains(s, "Chebyshev($(0..1))")
        @test contains(s, "(1+𝑪($(0..1), x))^1")
        @test contains(s, "(1-𝑪($(0..1), x))^2")
    end
    @testset "JacobiWeight" begin
        w = JacobiWeight(1.0, 0.0, Chebyshev())
        s = repr(w)
        @test contains(s, "Chebyshev()")
        @test contains(s, "(1+x)^1")

        w = JacobiWeight(0.0, 1.0, Chebyshev())
        s = repr(w)
        @test contains(s, "Chebyshev()")
        @test contains(s, "(1-x)^1")

        w = JacobiWeight(1.0, 1.0, Chebyshev())
        s = repr(w)
        @test contains(s, "Chebyshev()")
        @test contains(s, "(1-x^2)^1")

        w = JacobiWeight(1.0, 2.0, Chebyshev())
        s = repr(w)
        @test contains(s, "Chebyshev()")
        @test contains(s, "(1+x)^1")
        @test contains(s, "(1-x)^2")

        w = JacobiWeight(1,2,Chebyshev(0..1))
        s = repr(w)
        @test contains(s, "Chebyshev($(0..1))")
        @test contains(s, "(1+𝑪($(0..1), x))^1")
        @test contains(s, "(1-𝑪($(0..1), x))^2")
    end
end

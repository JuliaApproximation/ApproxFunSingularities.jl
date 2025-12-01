@testset "DiracDelta and PointSpace" begin
    a,b=DiracDelta(0.),DiracDelta(1.)
    f=Fun(exp)
    g=a+0.2b+f
    @test components(g)[2](0.) ≈ 1.
    @test g(.1) ≈ exp(.1)
    @test sum(g) ≈ (sum(f)+1.2)

    #Checks prevoius bug
    δ=DiracDelta()
    x=Fun()
    w=sqrt(1-x^2)
    @test (w+δ)(0.1) ≈ w(0.1)
    @test sum(w+δ) ≈ sum(w)+1

    ## PointSpace
    f=Fun(x->(x-0.1),PointSpace([0,0.1,1]))
    g = f + Fun(2..3)
    @test f(0.0) ≈ g(0.0) ≈ -0.1
    @test f(0.1) ≈ g(0.1) ≈ 0.0
    @test f(1.0) ≈ g(1.0) ≈ 0.9

    @test g(2.3) ≈ 2.3

    h = a + Fun(2..3)

    # for some reason this test is broken only on Travis
    @test_skip g/h ≈ f/a + Fun(1,2..3)
end

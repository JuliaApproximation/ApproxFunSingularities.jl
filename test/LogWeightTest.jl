@testset "LogWeight" begin
    x=Fun(identity,-1..1)
    f=exp(x+1)-1
    @test log(f)(0.1) ≈ log(f(0.1))

    x=Fun(identity,0..1)
    f=exp(x)-1
    @test log(f)(0.1) ≈ log(f(0.1))

    x=Fun(identity,0..1)
    @test Fun(exp(x)/x-1/x,Chebyshev)(0.1) ≈ (exp(0.1)-1)/0.1

    x=Fun(identity,0..1)
    f=1/x
    p=integrate(f)
    @test (p-p(1.))(0.5) ≈ log(0.5)

    f=1/(1-x)
    p=integrate(f)
    @test (p-p(0.))(0.5) ≈ -log(1-0.5)

    @testset "#393" begin
        x = Fun(0..1)
        @time f = exp(x)*sqrt(x)*log(1-x)
        @test f(0.1) ≈ exp(0.1)*sqrt(0.1)*log(1-0.1)
    end
end

@testset "Sqrt" begin
    x=Fun(identity);
    @time @test sqrt(cos(π/2*x))(.1) ≈ sqrt(cos(.1π/2))

    x=Fun(identity,-2..2)
    @time u=sqrt(4-x^2)/(2π)

    @test u(.1) ≈ sqrt(4-0.1^2)/(2π)
    @test sum(u) ≈ 1

    #this call threw an error, which we check
    @test length(values(u)) == 1

    f = Fun(x->x*cot(π*x/2))
    x = Fun(identity)
    u = Fun(JacobiWeight(1.,1.,ChebyshevInterval()), (f/(1-x^2)).coefficients)
    @test 1/(0.1*cot(π*0.1/2)) ≈ (1/u)(.1)

    @test (x/u)(.1) ≈ tan(π*.1/2)

    f=Fun(x->exp(-x^2),Line(0.,0.,-.5,-.5),400)
    @time @test sum(f) ≈ sqrt(π)

    f=Fun(x->exp(x)/sqrt(1-x.^2),JacobiWeight(-.5,-.5))
    @test f(.1) ≈ (x->exp(x)/sqrt(1-x.^2))(.1)

    @time @test norm(Fun(exp,Legendre(0..1))+sqrt(Fun(0..1))) ≈ 2.491141949903508

    @testset "sampling Chebyshev" begin
        x=Fun(identity)
        f = exp(x)/sqrt(1-x^2)
        @time g = cumsum(f)
        @test abs(g(-1)) ≤ 1E-15
        @test g'(0.1) ≈ f(0.1)
    end

    @testset "Complex domains sqrt" begin
        a=1+10*im; b=2-6*im
        d = IntervalCurve(Fun(x->1+a*x+b*x^2))

        x=Fun(d)
        w=sqrt(abs(leftendpoint(d)-x))*sqrt(abs(rightendpoint(d)-x))

        @test sum(w/(x-2.))/(2π*im) ≈ (-4.722196879007759+2.347910413861846im)
        @test linesum(w*log(abs(x-2.)))/π ≈ (88.5579588360686)

        a=Arc(0.,1.,0.,π/2)
        ζ=Fun(identity,a)
        f=Fun(exp,a)*sqrt(abs((ζ-1)*(ζ-im)))
    end

    @time @test norm(Fun(exp,Legendre(0..1))+sqrt(Fun(0..1))) ≈ 2.491141949903508
end

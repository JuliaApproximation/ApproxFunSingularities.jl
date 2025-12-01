@testset "Bessel" begin
    @time for ν in (1.,0.5,2.,3.5)
        println("        ν = $ν")
        S=JacobiWeight(-ν,0.,Chebyshev(0..1))
        D=Derivative(S)
        x=Fun(identity,domain(S))
        L=(x^2)*D^2+x*D+(x^2-ν^2);
        u=\([rdirichlet(S);rneumann(S);L],[bessely(ν,1.),.5*(bessely(ν-1.,1.)-bessely(ν+1.,1.)),0];
                    tolerance=1E-10)
        @test ≈(u(.1),bessely(ν,.1);atol=eps(1000000.)*max(abs(u(.1)),1))
        u=Fun(x->bessely(ν,x),S)
        @test ≈(u(.1),bessely(ν,.1);atol=eps(10000.)*max(abs(u(.1)),1))
        u=Fun(x->besselj(ν,x),S)
        @test ≈(u(.1),besselj(ν,.1);atol=eps(10000.)*max(abs(u(.1)),1))
    end

    @time for ν in (1.,0.5,0.123,3.5)
        println("        ν = $ν")
        S=JacobiWeight(ν,0.,Chebyshev(0..1))
        D=Derivative(S)
        x=Fun(identity,domain(S))
        L=(x^2)*D^2+x*D+(x^2-ν^2);

        u=\([rdirichlet(S);rneumann(S);L],[besselj(ν,1.),.5*(besselj(ν-1.,1.)-besselj(ν+1.,1.)),0];
                    tolerance=1E-10)
        @test ≈(u(.1),besselj(ν,.1);atol=eps(1000000.)*max(abs(u(.1)),1))
        u=Fun(x->besselj(ν,x),S)
        @test ≈(u(.1),besselj(ν,.1);atol=eps(10000.)*max(abs(u(.1)),1))
    end
end

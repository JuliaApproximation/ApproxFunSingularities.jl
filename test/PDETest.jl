@testset "PDE" begin
    @testset "Laplacian" begin
        S=WeightedJacobi(1,1)^2
        L=Laplacian(S)
        testbandedblockbandedoperator(L)
    end
    @testset "Zero Dirichlet" begin
        S = JacobiWeight(1.,1.,Jacobi(1.,1.))^2
        Δ = Laplacian(S)

        testbandedblockbandedoperator(Δ)

        u = Fun((x,y)->sin(π*x)*sin(π*y),S)
        f = -2π^2*u

        F = qr(Δ)
        ApproxFunBase.resizedata!(F,:,1000)
        @time v=F\f
        @test norm((u-v).coefficients)<100eps()


        F=qr(Δ)
        ApproxFunBase.resizedata!(F.R_cache,:,100)
        ApproxFunBase.resizedata!(F.R_cache,:,1000)
        @time v=F \ f
        @test norm((u-v).coefficients)<100eps()

        F=qr(Δ)
        @time v=F\f
        @test norm((u-v).coefficients)<100eps()
    end
    @testset "Gaussian zero Dirichlet Poisson" begin
        S=JacobiWeight(1.,1.,Jacobi(1.,1.))^2
        Δ=Laplacian(S)

        f=Fun((x,y)->exp(-10(x+.2)^2-20(y-.1)^2),rangespace(Δ))  #default is [-1,1]^2
        @time v = \(Δ,f;tolerance=1E-14)
        @test norm((Δ*v-f).coefficients) < 1E-14

    end
    @testset "check we dispatch correctly to get fast build" begin
        S = JacobiWeight(1.,1.,Jacobi(1.,1.))^2
        Δ = Laplacian(S)
        @time S = view(Δ.op.ops[1].ops[1].op,Block.(1:40), Block.(1:40))
        @test typeof(S.parent.domaintensorizer) == ApproxFunBase.Trivial2DTensorizer
    end

    @testset "Operator resize" begin
        S=ChebyshevDirichlet()^2
        B=Dirichlet(S)
        f = Fun((x,y)->exp(x)*sin(y),S)
        @test norm((Fun((x,y)->exp(x)*sin(y),∂(domain(S))) - B*f).coefficients) < 100eps()


        S=JacobiWeight(1.,1.,Jacobi(1.,1.))^2
        Δ=Laplacian(S)

        @test cache(Δ)[1:100,1:100]  ≈ Δ[1:100,1:100]
        @test cache(Δ;padding=true)[1:100,1:100]  ≈ Δ[1:100,1:100]

        @test cache(Δ)[5:100,7:100]  ≈ Δ[5:100,7:100]
        @test cache(Δ;padding=true)[5:100,7:100]  ≈ Δ[5:100,7:100]

        # Check that QR is growing correctly
        for col in (1,2,3,10,11,40)
            QR=qr(Δ)
            resizedata!(QR.R_cache,:,col+100)
            resizedata!(QR,:,col)
            QR2=qr!(CachedOperator(RaggedMatrix,Δ;padding=true))
            resizedata!(QR2.R_cache,:,QR.ncols+100)
            resizedata!(QR2,:,QR.ncols)
            n=min(size(QR.H,1),size(QR2.H,1))
            @test QR.H[1:n,1:col] ≈ QR2.H[1:n,1:col]
            @test QR.R_cache[1:col,1:col] ≈ QR2.R_cache[1:col,1:col]
            @test QR.R_cache[1:col+10,1:col+10] ≈ QR2.R_cache[1:col+10,1:col+10]
        end

        QRR=qr(Δ)
        QR2=qr!(CachedOperator(RaggedMatrix,Δ;padding=true))
        for col in (80,200)
            resizedata!(QRR,:,col)
            resizedata!(QR2,:,QRR.ncols)
            n=min(size(QRR.H,1),size(QR2.H,1))
            @test QRR.H[1:n,1:col] ≈ QR2.H[1:n,1:col]
            @test QRR.R_cache[1:col,1:col] ≈ QR2.R_cache[1:col,1:col]
            @test QRR.R_cache[1:col+10,1:col+10] ≈ QR2.R_cache[1:col+10,1:col+10]
        end

        # this checks a bug
        QRR=qr(Δ)
        resizedata!(QRR,:,548)
        resizedata!(QRR,:,430)


        u=Fun((x,y)->sin(π*x)*sin(π*y),S)
        f=-2π^2*u


        QRR=qr(Δ)
        v=QRR\f
        @test norm((u-v).coefficients)<100eps()

        v=Δ\f
        @test norm((u-v).coefficients)<100eps()


        f=Fun((x,y)->exp(-10(x+.2)^2-20(y-.1)^2),rangespace(Δ))  #default is [-1,1]^2
        @time v=\(Δ,f;tolerance=1E-14)
        @test norm((Δ*v-f).coefficients)<1E-14

        KO=Δ.op.ops[1].ops[1].op

        M=BandedBlockBandedMatrix(view(KO,1:4,1:4))
        @test norm(BandedBlockBandedMatrix(view(KO,1:4,2:4))-M[:,2:4]) < 10eps()
        @test norm(BandedBlockBandedMatrix(view(KO,1:4,3:4))-M[:,3:4]) < 10eps()
    end
end
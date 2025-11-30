@testset "Speed test" begin
    S = JacobiWeight(1.,1.,Jacobi(1.,1.))^2
    Δ = Laplacian(S)

    f = Fun((x,y)->sin(π*x)*sin(π*y),S)

    QR1=qr(Δ)
    ApproxFunBase.resizedata!(QR1,:,400)
        \(QR1,f; tolerance=1E-10)
    QR1=qr(Δ)
        @time Δ[Block.(1:40), Block.(1:40)]
        @time ApproxFunBase.resizedata!(QR1,:,400)
        @time \(QR1,f; tolerance=1E-10)
    println("Laplace Dirichlet: should be ~0.015, 0.015, 0.001")
end

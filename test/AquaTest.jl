using Aqua
@testset "Project quality" begin
    Aqua.test_all(ApproxFunSingularities, ambiguities=false,
        stale_deps=(; ignore=[:ApproxFunBaseTest]), piracies = false)
end

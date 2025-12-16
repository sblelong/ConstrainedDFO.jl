using Test
using ConstrainedDFO
using Manopt

@testset "Stopping Criteria" begin
    h(p) = [sum(p .^ 2) - 4]
    M = EqualityManifold(h, 2, 3)
    p = [2.0, 0.0, 0.0]
    f(p) = sum(p)
    mco = ManifoldCostObjective(f)
    pb = DefaultManoptProblem(M, mco)
    sc1 = StopAfterEvaluation(10000)
    em = FractionEvalManager(10000, 0.1)
    @test get_reason(sc1) == ""
    s = RDFOState(M, p, sc1, ProjectionRetraction())
    @test !sc1(pb, s, 0, 0)
    @test sc1(pb, s, 0, 10000)

    sc2 = StopWhenWithinRadius()
    @test get_reason(sc2) == ""
end

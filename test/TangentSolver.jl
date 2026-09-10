@testset "Tangent Solver" begin
    M = Sphere(2)
    p1 = [1.0, 0.0, 0.0]
    cost(M, p) = sum(p)
    mco = ManifoldCostObjective(cost)
    MTS = MADSTangentSolver()
    radius1 = invertibility_radius(M, p1)
    solve!(
        MTS,
        mco,
        M,
        p1,
        StabilizedRetraction(),
        radius1,
        0
    )
    @test get_radius_evaluation(MTS) > 0
    @test length(get_data_d(MTS)) == 10
    @test length(get_data_f(MTS)) == 10
    @test length(get_data_g(MTS)) == 0
    @test length(get_data_Rpv(MTS)) == 10
    p2 = get_data_Rpv(MTS)[8]
    radius2 = invertibility_radius(M, p2)
    solve!(
        MTS,
        mco,
        M,
        p2,
        StabilizedRetraction(),
        radius2,
        0
    )
    @test length(get_data_f(MTS)) == 221
    @test isapprox(get_data_f(MTS)[end], -sqrt(3); atol = 1.0e-8)
end

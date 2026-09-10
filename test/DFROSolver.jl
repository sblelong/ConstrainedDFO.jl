@testset "DFRO Solver" begin
    M1 = Sphere(2)
    f1(p) = p[1]
    g1(p) = []
    p1 = [1.0, 0.0, 0.0]
    res1 = DFROSolver(M1, f1, g1, p1, 0; print_level = 0)

    @test is_point(M1, res1)
    @test isapprox(f1(res1), -1.0)

    res2 = DFROSolver(M1, f1, g1, p1, 0; retraction_method = ProjectionRetraction(), print_level = 0)

    @test is_point(M1, res2)
    @test isapprox(f1(res2), -1.0)

    h2(p) = [sum(p .^ 2) - 1] # Don't use norm(p)^2 because the operator is not supported by MathOptInterface
    M2 = EqualityManifold(h2, 2, 3)
    res3 = DFROSolver(M2, f1, g1, p1, 0; print_level = 0)

    @test is_point(M2, res3)
    @test isapprox(f1(res3), -1.0)

    # On a parabola
    h3(p) = [p[1]^2 - p[2]]
    M3 = EqualityManifold(h3, 1, 2)
    p2 = [2.0, 4.0]
    f2(p) = p[2]
    res4 = DFROSolver(M3, f2, g1, p2, 0; print_level = 0)

    @test is_point(M3, res4)
    @test isapprox(f1(res4), 0.0; atol = 1.0e-4)

    # With Rosenbrock
    f3(p) = 100 * (p[2] - p[1]^2)^2 + (1 - p[1])^2 + 100 * (p[3] - p[2]^2)^2 + (1 - p[2])^2
    res5 = DFROSolver(M1, f3, g1, p1, 0; print_level = 0)

    @test is_point(M1, res5)
    @test f3(res5) < 2.45e-1
end

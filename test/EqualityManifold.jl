using Test
using ConstrainedDFO
using LinearAlgebra

@testset "Equality Manifolds" begin
    h1(p) = [sum(p .^ 2) - 4]
    M1 = EqualityManifold(h1, 2)

    @test manifold_dimension(M1) == 2
    @test representation_size(M1) == (3,)

    p1 = [2.0, 0.0, 0.0]
    p2 = [2.1, 0.0, 0.0]
    p3 = [2.0, 2.0]

    @test is_point(M1, p1)
    @test !is_point(M1, p2)
    @test !is_point(M1, p3)

    X1 = [0.0, 1.0, 1.0]
    X2 = [0.0, 0.0]

    @test is_vector(M1, p1, X1)
    @test !is_vector(M1, p1, X2)
    @test !is_vector(M1, p2, X1)
    @test !is_vector(M1, p2, X2)

    B1 = get_basis(M1, p1, DefaultOrthonormalBasis())
    @test transpose(B1) * B1 == I # Check that the given basis is indeed orthonormal
    X3 = 3 .* B1[:, 1]
    X4 = -3 .* B1[:, 2]
    X5 = X3 .+ X4
    @test is_vector(M1, p1, X3)
    @test is_vector(M1, p1, X4)
    @test is_vector(M1, p1, X5)

    h2(p) = [p[1]^2 - p[2]^3]
    M2 = EqualityManifold(h2, 1)
    p4 = [1.0, 1.0]
    p5 = [0.1, (0.01)^(1 / 3)]
    p6 = [0.0, 0.0]

    X6 = [3.0, 2.0]
    X7 = [3.001, 2.0]
    X8 = [3 * (0.01)^(2 / 3), 0.2]
    @test is_point(M2, p4; atol = 1.0e-12)
    @test is_point(M2, p5; atol = 1.0e-12)
    @test is_vector(M2, p4, X6; atol = 1.0e-12)
    @test !is_vector(M2, p4, X7; atol = 1.0e-12)
    @test is_vector(M2, p5, X8; atol = 1.0e-12)

    @test_throws Any get_basis(M2, p6, DefaultOrthonormalBasis())

end

using Test
using ConstrainedDFO

@testset "Equality Manifolds" begin
    h(p) = [sum(p .^ 2) - 4]
    M = EqualityManifold(h, 2)

    @test manifold_dimension(M) == 2
    @test representation_size(M) == (3,)

    p1 = [2.0, 0.0, 0.0]
    p2 = [2.1, 0.0, 0.0]
    p3 = [2.0, 2.0]

    @test is_point(M, p1)
    @test !is_point(M, p2)
    @test !is_point(M, p3)

    X1 = [0.0, 1.0, 1.0]
    X2 = [0.0, 0.0]

    @test is_vector(M, p1, X1)
    @test !is_vector(M, p1, X2)
    @test !is_vector(M, p2, X1)
    @test !is_vector(M, p2, X2)
end

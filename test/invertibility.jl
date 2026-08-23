@testset "Invertibility radii" begin
    M1 = Sphere(2)
    p1 = [0.0, 1.0, 0.0]
    @test invertibility_radius(M1, p1) == π

    h(p) = norm(p)^2 - 1
    M2 = EqualityManifold(h, 1, 2)
    p2 = [1.0, 0.0]
    @test invertibility_radius(M2, p2) == 1
    @test invertibility_radius(M2, p2; ρ = OneOverSpectral()) == 0.5
    @test invertibility_radius(M2, p2; ρ = OneOverSqrtSpectral()) == 1 / sqrt(2)
    @test invertibility_radius(M2, p2; ρ = NOverSqrtSpectral()) == 2 / sqrt(2)
end

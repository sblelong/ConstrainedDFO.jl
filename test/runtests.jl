using Test
using ConstrainedDFO
using LinearAlgebra
using ManifoldsBase
using Manifolds
using Manopt:
    ManifoldCostObjective

@testset "ConstrainedDFO.jl" begin
    include("EqualityManifold.jl")
    include("StoppingCriteria.jl")
    include("TangentSolver.jl")
end

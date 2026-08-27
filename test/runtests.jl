using Test
using ConstrainedDFO
using LinearAlgebra
using ManifoldsBase
using Manifolds
using Manopt:
    ManifoldCostObjective

@testset "ConstrainedDFO.jl" begin
    include("BlackboxProblem.jl")
    include("EqualityManifold.jl")
    include("StoppingCriteria.jl")
    include("TangentSolver.jl")
    include("invertibility.jl")
end

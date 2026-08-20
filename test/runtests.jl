using Test
using ConstrainedDFO
using ManifoldsBase

@testset "ConstrainedDFO.jl" begin
    include("EqualityManifold.jl")
    include("StoppingCriteria.jl")
end

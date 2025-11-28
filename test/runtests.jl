using ConstrainedDFO, ManifoldsBase
using Test

@testset "ConstrainedDFO.jl" begin
    include("EqualityManifold.jl")
    include("StoppingCriteria.jl")
end

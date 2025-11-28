using Test
using ConstrainedDFO
using Manopt

@testset "Stopping Criteria" begin
    c1 = StopAfterEvaluation(10000)
    @test get_reason(c1) == ""

    c2 = StopWhenWithinRadius()
    @test get_reason(c2) == ""
end

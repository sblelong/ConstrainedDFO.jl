using Test
using ConstrainedDFO

@testset "Evaluation Manager" begin
    em = FractionEvalManager(10000, 0.1)
    @test get_eval_budget(em) == 1000
    update_remaining_evals!(em, 1000)
    @test get_remaining_evals(em) == 9000
end

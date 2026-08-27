using NLPModels

@testset "BlackboxProblem" begin
    BP = BlackboxProblem(2, 1, x -> x[1] + x[2], x -> [x[1] - 1])
    @test get_dimension(BP) == 2
    @test get_n_ineqs(BP) == 0
    @test get_n_eqs(BP) == 1
    
    x = [1.0, -1.0]
    @test eval_objective(BP, x) == 0.0
    @test eval_ineqs(BP, x) == Float64[]
    @test eval_eqs(BP, x) == [0.0]
end

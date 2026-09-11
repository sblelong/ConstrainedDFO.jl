"""
    BlackboxProblem
"""
mutable struct BlackboxProblem
    n::Int
    p::Int
    m::Int
    f::Function
    h::Function
    g::Function
end

BlackboxProblem(n, p, f, h) = BlackboxProblem(n, p, 0, f, h, x -> Float64[])

get_dimension(BP::BlackboxProblem) = BP.n
get_n_ineqs(BP::BlackboxProblem) = BP.m
get_n_eqs(BP::BlackboxProblem) = BP.p

eval_objective(BP::BlackboxProblem, x) = BP.f(x)
eval_ineqs(BP::BlackboxProblem, x) = BP.g(x)
eval_eqs(BP::BlackboxProblem, x) = BP.h(x)

"""
    BlackboxInstance
"""
mutable struct BlackboxInstance
    problem::BlackboxProblem
    x0::Vector{Float64}
end

get_dimension(BI::BlackboxInstance) = get_dimension(BI.problem)
get_n_ineqs(BI::BlackboxInstance) = get_n_ineqs(BI.problem)
get_n_eqs(BI::BlackboxInstance) = get_n_eqs(BI.problem)

eval_objective(BI::BlackboxInstance, x) = eval_objective(BI.problem, x)
eval_ineqs(BI::BlackboxInstance, x) = eval_ineqs(BI.problem, x)
eval_eqs(BI::BlackboxInstance, x) = eval_eqs(BI.problem, x)

get_x0(BI::BlackboxInstance) = BI.x0

"""
    BlackboxProblem
"""
mutable struct BlackboxProblem
    n::Int
    p::Int
    m::Int
    f
    h
    g
    lb::Vector{Float64}
    ub::Vector{Float64}

    function BlackboxProblem(n::Int, p::Int, m::Int, f, h, g, lb::Vector{Float64}, ub::Vector{Float64})
        length(lb) == n || throw(DimensionMismatch("Incoherent size of the lower bound vector while instantiating BlackboxProblem: $(length(lb))≠ $(n)."))
        length(ub) == n || throw(DimensionMismatch("Incoherent size of the upper bound vector while instantiating BlackboxProblem: $(length(ub))≠ $(n)."))
        return new(n, p, m, f, h, g, lb, ub)
    end
end

BlackboxProblem(n::Int, p::Int, m::Int, f, h, g; lb::Vector{Float64} = fill(typemin(Float64), n), ub::Vector{Float64} = fill(typemax(Float64, n))) = BlackboxProblem(n, p, m, f, h, g, lb, ub)
BlackboxProblem(n::Int, p::Int, f, h) = BlackboxProblem(n, p, 0, f, h, x -> Float64[])

get_dimension(BP::BlackboxProblem) = BP.n
get_n_ineqs(BP::BlackboxProblem) = BP.m
get_n_eqs(BP::BlackboxProblem) = BP.p
get_lbounds(BP::BlackboxProblem) = BP.lb
get_ubounds(BP::BlacboxProblem) = BP.ub

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
get_lbounds(BI::BlackboxInstance) = get_lbounds(BI.problem)
get_ubounds(BI::BlackboxInstance) = get_ubounds(BI.problem)

eval_objective(BI::BlackboxInstance, x) = eval_objective(BI.problem, x)
eval_ineqs(BI::BlackboxInstance, x) = eval_ineqs(BI.problem, x)
eval_eqs(BI::BlackboxInstance, x) = eval_eqs(BI.problem, x)

get_x0(BI::BlackboxInstance) = BI.x0

using ConstrainedDFO
using Manifolds, Manopt
using LinearAlgebra
using Printf

"""Thrown by the cost function once the budget of evaluations is exhausted, to stop the solver."""
struct BudgetExhausted <: Exception end

"""
Add a unit sphere constraint to an instance and solve it with the MADS solver implemented in Manopt.
The budget `max_evals` counts the evaluations of distinct points: Manopt re-evaluates its current point at each iteration,
which is served from a cache instead, without being logged nor counted. The solver is stopped as soon as the budget is exhausted,
in which case `nothing` is returned (the results are read from the logs).
"""
function solve_sphere_manopt(BI::BlackboxInstance; max_evals::Int = 200 * (get_dimension(BI) + 1), tol_eqs::Float64 = 1.0e-8)
    dimension = get_dimension(BI)
    M = Manifolds.Sphere(dimension - 1)

    x0 = ConstrainedDFO.get_x0(BI)
    p0 = norm(x0) == 0 ? [[1.0] ; [0.0 for _ in 1:(dimension - 1)]] : x0 ./ norm(x0)

    # Values of the points already evaluated, whose number is the number of evaluations
    evaluated = Dict{Vector{Float64}, Float64}()
    function f(M::Manifolds.Sphere, p)
        haskey(evaluated, p) && return evaluated[p]
        length(evaluated) ≥ max_evals && throw(BudgetExhausted())
        hval = norm(p)^2 - 1.0
        fval = abs(hval) ≤ tol_eqs ? eval_objective(BI, p) : ConstrainedDFO.FAILURE_MAX
        line = @sprintf("%-30.15g%-30.15g", fval, hval)
        println(line)
        evaluated[Vector{Float64}(p)] = fval
        return fval
    end

    # The budget is enforced by f. The iteration cap only guards against iterations that would keep revisiting evaluated points.
    stopping_criterion = StopAfterIteration(10 * max_evals) | StopWhenPollSizeLess(1.0e-10)

    try
        return mesh_adaptive_direct_search(M, f, p0; stopping_criterion = stopping_criterion)
    catch e
        e isa BudgetExhausted || rethrow()
        return nothing
    end
end

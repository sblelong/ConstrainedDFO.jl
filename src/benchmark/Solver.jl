using ManifoldsBase

"""
    Test
"""
solve_problem(::AbstractDFSolver, ::AbstractBenchmarkProblem)

####################################################################
# rDFO
####################################################################

mutable struct RDFOSolver <: AbstractDFSolver end

function solve_problem(::RDFOSolver, BPE::BenchmarkEqualityProblem; max_evals::Int = 1000 * (get_dimension(BPE) + 1), invertibility_bound::AbstractInvertibilityBound = default_invertibility_bound(get_equality_manifold(BPE), default_retraction_method(get_equality_manifold(BPE))))
    n = get_dimension(BPE)
    x0 = get_x0(BPE)

    h(x) = eval_eqs(BPE, x)
    h0 = h(x0)
    p = length(h0)
    M = EqualityManifold(h, n - p, n)

    f(M::EqualityManifold, x) = eval_obj(BPE, x)
    g(x) = eval_ineqs(BPE, x)

    if length(g(x0)) > 0
        return rDFO(M, f, x0; max_evals = max_evals, invertibility_bound = invertibility_bound, inequality_constraints = g)
    else
        return rDFO(M, f, x0; max_evals = max_evals, invertibility_bound = invertibility_bound)
    end
end

function solve_problem(::RDFOSolver, BPM::BenchmarkManifoldProblem; max_evals::Int = 1000 * get_dimension(BPM), kwargs...)
    n = get_dimension(BPM)
    x0 = get_x0(BPM)

    f(M::AbstractManifold, x) = eval_obj(BPM, x)

    result, eval_data, iterates_history, objective_history, vs = rDFO(get_equality_manifold(BPM), f, x0; max_evals = max_evals)

    return rDFO(get_equality_manifold(BPM), f, x0; max_evals = max_evals)
end

export RDFOSolver, solve_problem

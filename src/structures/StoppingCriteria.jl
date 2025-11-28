using Manopt

export StopAfterEvaluation, StopWhenInsideRadius

"""
    StopAfterEvaluation <: StoppingCriterion

A functor for a stopping criterion to stop after a maximal number of blackbox evaluations. Fields and constructor are the same as `StopAfterIteration`.

TODO.
"""
mutable struct StopAfterEvaluation <: StoppingCriterion
    max_evals::Int
    at_eval::Int
end

StopAfterEvaluation(max_evals::Int) = StopAfterEvaluation(max_evals, 0)

function (c::StopAfterEvaluation)(
        ::P, ::S, n_evals::Int
    ) where {P <: AbstractManoptProblem, S <: AbstractManoptSolverState}
    if n_evals == 0
        c.at_eval = 0
    end
    if n_evals ≥ c.max_evals
        c.at_eval = n_evals
        return true
    end
    return false
end

function get_reason(c::StopAfterEvaluation)
    if c.at_eval ≥ c.max_evals
        return "The algorithm has reached its maximal number of blackbox evaluations (i.e. $(c.max_evals))"
    end
    return ""
end

mutable struct StopWhenInsideRadius <: StoppingCriterion end

function (c::StopWhenInsideRadius)(
        mp::AbstractManoptProblem, s::AbstractManoptSolverState, norm_d
    )
    M = get_manifold(mp)
    p = get_iterate(s)
    return norm_d ≤ injectivity_bound(M, p, ProjectionRetraction())
end

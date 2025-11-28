using Manopt

import Manopt: get_reason
export StopAfterEvaluation, StopWhenWithinRadius

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

mutable struct StopWhenWithinRadius <: StoppingCriterion
    at_iteration::Int
    radius::Float64
end

StopWhenWithinRadius() = StopWhenWithinRadius(-1, typemax(Float64))

function (c::StopWhenWithinRadius)(
        mp::AbstractManoptProblem, s::AbstractManoptSolverState, norm_d, k::Int
    )
    M = get_manifold(mp)
    p = get_iterate(s)
    inj = injectivity_bound(M, p, ProjectionRetraction())
    if norm_d ≤ inj
        c.at_iteration = k
        c.radius = inj
        return true
    end
    return false
end

function get_reason(c::StopWhenWithinRadius)
    if c.at_iteration > 1
        return "A subproblem has been solved within the injectivity radius (bound: $(c.radius)) at iteration $(c.at_iteration)"
    end
    return ""
end

using Manopt
using LinearAlgebra

import Manopt: get_reason, stop_solver!
export DFStoppingCriterion, StopAfterEvaluation, StopWhenWithinRadius, StopRadiusAndBudget

abstract type DFStoppingCriterion <: StoppingCriterion end

"""
    StopAfterEvaluation <: DFStoppingCriterion

A functor for a stopping criterion to stop after a maximal number of blackbox evaluations. Fields and constructor are the same as `StopAfterIteration`.

TODO.
"""
mutable struct StopAfterEvaluation <: DFStoppingCriterion
    max_evals::Int
    at_iteration::Int
    at_eval::Int
end

StopAfterEvaluation(max_evals::Int) = StopAfterEvaluation(max_evals, -1, -1)

function (c::StopAfterEvaluation)(
        ::P, ::S, k::Int, n_evals::Int
    ) where {P <: AbstractManoptProblem, S <: AbstractManoptSolverState}
    if n_evals == 0
        c.at_iteration = -1
        c.at_eval = -1
    end
    if n_evals ≥ c.max_evals
        c.at_iteration = k
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

mutable struct StopWhenWithinRadius <: DFStoppingCriterion
    radius::Float64
    at_iteration::Int
    at_eval::Int
end

StopWhenWithinRadius() = StopWhenWithinRadius(typemax(Float64), -1, -1)

function (c::StopWhenWithinRadius)(
        mp::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int, n_evals::Int
    )
    M = get_manifold(mp)
    p = get_iterate(s)
    d = get_tangent_iterate(s)
    inj = injectivity_radius_bound(M, p, ProjectionRetraction())
    if norm(d) ≤ inj
        c.at_iteration = k
        c.radius = inj
        return true
    end
    return k == 3
end

function get_reason(c::StopWhenWithinRadius)
    if c.at_iteration > 1
        return "A subproblem has been solved within the injectivity radius (bound: $(c.radius)) at iteration $(c.at_iteration)"
    end
    return ""
end

mutable struct StopRadiusAndBudget <: DFStoppingCriterion
    radius::Float64
    max_evals::Int
    at_iteration::Int
    at_eval::Int
end

StopRadiusAndBudget(max_evals::Int) = StopRadiusAndBudget(typemax(Float64), max_evals, -1, -1)

function (c::StopRadiusAndBudget)(
        mp::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int, n_evals::Int, flag::Bool
    )
    if flag
        return n_evals ≥ c.max_evals
    end
    n_evals ≥ c.max_evals && return true
    M = get_manifold(mp)
    p = get_iterate(s)
    d = get_tangent_iterate(s)
    inj = injectivity_radius(M, p, ProjectionRetraction())
    if norm(d) ≤ inj
        c.at_iteration = k
        c.radius = inj
        return true
    end
    return false
end

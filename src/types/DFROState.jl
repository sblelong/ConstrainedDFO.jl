"""
    DFROState <: AbstractManoptSolverState

[`AbstractManoptSolverState`](@extref Manopt.AbstractManoptSolverState) dedicated to the RDFO solver.

# Fields
* `p` is the current iterate on the manifold.
* `d` is the current best tangent vector found at ``T_p\\mathcal{M}``.
"""
mutable struct DFROState{P, SC <: StoppingCriterion} <: AbstractManoptSolverState
    p::P
    d::P
    stop::SC
end

function DFROState(
        M::AbstractManifold,
        p::P,
        stopping_criterion::SC = StopWhenWithinRadius(),
        retraction_method::AbstractRetractionMethod = default_retraction_method(M)
    ) where {
        P,
        SC <: DFStoppingCriterion,
    }
    return DFROState{P, SC}(p, zeros(representation_size(M)), stopping_criterion)
end

set_iterate!(s::DFROState, p) = s.p = p
set_tangent_iterate!(s::DFROState, d) = s.d = d

get_iterate(s::DFROState) = s.p
get_tangent_iterate(s::DFROState) = s.d

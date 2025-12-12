using Manopt

export RDFOState

"""
# Fields
* `p` is the current iterate on the manifold.
* `d` is the current best tangent vector found at ``T_p\\mathcal{M}``.
"""
mutable struct RDFOState{P, SC <: StoppingCriterion} <: AbstractManoptSolverState
    p::P
    d::P
    stop::SC
end

function RDFOState(
        M::AbstractManifold,
        p::P,
        stopping_criterion::SC = StopWhenWithinRadius(),
        retraction_method::AbstractRetractionMethod = default_retraction_method(M)
    ) where {
        P,
        SC <: DFStoppingCriterion,
    }
    return RDFOState{P, SC}(p, zeros(representation_size(M)), stopping_criterion)
end

get_iterate(s::RDFOState) = s.p

get_tangent_iterate(s::RDFOState) = s.d

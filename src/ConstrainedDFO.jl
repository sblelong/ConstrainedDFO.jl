@doc """
    ConstrainedDFO.jl: derivative-free optimization under constraints.

- Documentation: [https://sblelong.github.io/ConstrainedDFO.jl/dev/](https://sblelong.github.io/ConstrainedDFO.jl/dev/)
- Repository: [https://github.com/sblelong/ConstrainedDFO.jl](https://github.com/sblelong/ConstrainedDFO.jl)
- Issues: [https://github.com/sblelong/ConstrainedDFO.jl/issues](https://github.com/sblelong/ConstrainedDFO.jl/issues)
"""
module ConstrainedDFO

import ManifoldsBase:
    check_size,
    check_point,
    check_vector,
    default_basis,
    default_retraction_method,
    exp!,
    get_basis,
    get_basis_orthonormal,
    get_coordinates_orthonormal,
    get_vector_orthonormal!,
    get_embedding,
    manifold_dimension,
    representation_size,
    retract_project!
import Manopt:
    get_reason,
    stop_solver!

using ForwardDiff:
    jacobian,
    hessian
using Ipopt
using JuMP
using LinearAlgebra
using Manifolds
using ManifoldsBase
using Manopt:
    AbstractManoptProblem,
    AbstractManoptSolverState
using NOMAD
using Random
using ResumableFunctions

include("types/EqualityManifold.jl")
include("types/EvalManager.jl")
include("types/RDFOState.jl")
include("types/ScaledSphere.jl")
include("types/StoppingCriteria.jl")

export AbstractEvalManager,
    AbstractInvertibilityBound,
    DFStoppingCriterion,
    EqualityManifold,
    FractionEvalManager,
    NOverSqrtSpectral,
    NOverSpectral,
    OneOverSqrtSpectral,
    OneOverSpectral,
    RDFOState,
    ScaledSphere,
    StopAfterEvaluation,
    StopRadiusAndBudget,
    StopWhenWithinRadius

export get_eval_budget,
    get_iterate,
    get_radius,
    get_remaining_evals,
    get_tangent_iterate,
    invertibility_radius,
    set_iterate!,
    set_tangent_iterate!,
    update_remaining_evals!

include("solvers/rdfo.jl")
include("solvers/tangent_solvers/DFSolver.jl")
include("solvers/tangent_solvers/mads.jl")

export AbstractDFRSolver,
    AbstractDFSolver,
    MADSDFRSolver

export process_details,
    solve!

include("utils/latin_hypercube_sampling.jl")
include("utils/redirect.jl")
include("utils/spherical_coordinates.jl")

export latin_hypercube_sampling,
    redirect_to_files,
    spherical_to_cartesian


end

"""
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
    stop_solver!,
    StoppingCriterion

using ForwardDiff:
    jacobian,
    hessian
using Ipopt
using JuMP
using LinearAlgebra
using Manifolds
using ManifoldsBase
using Manopt:
    AbstractManifoldCostObjective,
    AbstractManoptProblem,
    AbstractManoptSolverState
using NOMAD
using Random
using ResumableFunctions

# Riemannian submanifolds of ℝ^n defined as feasible sets for equality constraints
include("types/EqualityManifold.jl")
export AbstractInvertibilityBound,
    EqualityManifold,
    NOverSqrtSpectral,
    NOverSpectral,
    OneOverSqrtSpectral,
    OneOverSpectral
export invertibility_radius

# Scaled sphere, this structure is useful for comparison against parametrization
include("types/ScaledSphere.jl")
export ScaledSphere

# Stopping criteria for derivative-free solvers
include("types/StoppingCriteria.jl")
export DFStoppingCriterion,
    StopAfterEvaluation,
    StopRadiusAndBudget,
    StopWhenWithinRadius

include("types/EvalManager.jl")
export AbstractEvalManager,
    FractionEvalManager
export get_eval_budget,
    get_remaining_evals,
    update_remaining_evals!

include("types/DFROState.jl")
export DFROState
export
    get_iterate,
    get_radius,
    get_tangent_iterate,
    set_iterate!,
    set_tangent_iterate!

# Solvers
include("solvers/tangent_solvers/TangentSolver.jl")
export AbstractTangentSolver
include("solvers/tangent_solvers/mads.jl")
export MADSDFRSolver
export process_details,
    solve!

include("solvers/DFROSolver.jl")
export DFROSolver

include("utils/latin_hypercube_sampling.jl")
include("utils/redirect.jl")
include("utils/spherical_coordinates.jl")

export latin_hypercube_sampling,
    redirect_to_files,
    spherical_to_cartesian

end

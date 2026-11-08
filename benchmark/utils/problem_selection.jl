using CUTEst, NLPModels
using Random
using ConstrainedDFO

include(joinpath(@__DIR__, "project.jl"))

"""Seed of the random generator, set before computing each first point and before each solver call, so that every run starts
from the same random state whatever ran before it (the projections onto the equality constraints perturb their starting point randomly when they fail)."""
const SEED = 0

"""Whether a problem passes `filter` once loaded, the model being finalized in any case (CUTEst only allows one open model)."""
function model_filter(filter, problem_name::String)
    nlp = CUTEstModel(problem_name)
    try
        return filter(nlp)
    finally
        finalize(nlp)
    end
end

"""Metadata pre-selection: at least one equality constraint and more free variables than equality constraints."""
has_free_dimensions(meta) = meta["constraints"]["equality"] > 0 &&
    meta["variables"]["number"] - meta["variables"]["fixed"] > meta["constraints"]["equality"]

"""
CUTEst problems of the general benchmarks (relaxable, unrelaxable and DFRO radii): at most 100 variables,
at least one nonlinear equality constraint, inequality constraints and bounds possible.
"""
function general_problems()
    CUTEst.set_mastsif()
    metadata_filter(meta) = has_free_dimensions(meta) && meta["constraints"]["nonlinear"] > 0
    problems_names = CUTEst.select_sif_problems(
        max_var = 100,
        max_con = 100,
        min_con = 1,
        contype = [6, 7],
        custom_filter = metadata_filter
    )
    # The metadata does not tell which constraints are linear.
    filter!(name -> model_filter(nlp -> !isempty(setdiff(nlp.meta.jfix, nlp.meta.lin)), name), problems_names)

    to_exclude = [
        # Can't find a feasible first guess
        "SSINE",
        "VANDANIUMS",
        "TRO6X2",

        # Can't compute invertibility radius: Hessian contains Inf or NaN
        "BA-L1",
        "CSFI1",
        "CSFI2",
        "LAKES",
        "STRTCHDVNE",
    ]
    return filter!(name -> name ∉ to_exclude, problems_names)
end

"""
CUTEst problems of the linear equalities benchmark: at most 100 variables and 100 constraints,
all equality constraints linear, inequality constraints of any kind and bounds possible.
"""
function linear_equality_problems()
    CUTEst.set_mastsif()
    problems_names = CUTEst.select_sif_problems(
        max_var = 100,
        max_con = 100,
        custom_filter = has_free_dimensions
    )
    # The metadata does not tell which constraints are linear.
    return filter!(name -> model_filter(nlp -> nlp.meta.jfix ⊆ nlp.meta.lin, name), problems_names)
end

function unconstrained_problems()
    CUTEst.set_mastsif()
    problems_names = CUTEst.select_sif_problems(
        max_var = 100,
        contype = :unc
    )

    to_exclude = [
        "MUONSINELS", # Dimension 1, trivial.
        "LUKSAN22LC", # Not included in the MATLAB version of CUTEst - RDSE-DD cannot solve it.
    ]
    return filter!(e -> e ∉ to_exclude, problems_names)
end

"""
First point shared by all solvers on an instance: the projection of its starting point onto the equality constraints and bounds.
Returns `nothing` when the problem must be skipped, i.e. when no such point is found, or when it violates an equality constraint by
more than `tol_eqs`. The check is the same in every setting, so that relaxable and unrelaxable benchmarks run on the same problems.
The random generator is seeded first, since the projection perturbs its starting point randomly when it fails.
"""
function feasible_first_point(BI::BlackboxInstance; tol_eqs::Float64 = 1.0e-8, seed::Int = SEED)
    Random.seed!(seed)
    x0f = try
        make_x0_feasible(BI)
    catch e
        println("Couldn't find a feasible first point. Skipping.")
        return nothing
    end
    if any(abs.(eval_eqs(BI, x0f)) .> tol_eqs)
        println("The first point violates the equality constraints by more than $(tol_eqs). Skipping.")
        return nothing
    end
    return x0f
end

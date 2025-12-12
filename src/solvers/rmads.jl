using ManifoldsBase
using Manopt
using NOMAD
using Printf

export rDFO, local_bb_wrapper

function local_bb_wrapper(mco::AbstractManifoldCostObjective, M::AbstractManifold, p, v)
    global n_evals, eval_data
    d = get_vector(M, p, v)
    Pd = retract(M, p, d)
    fd = get_cost(M, mco, Pd)
    n_evals += 1
    # norm(v) ≤ 1.0e-8 && println("Small v. Linked gap: $(distance(M, Pd, p)). Eval nb: $(n_evals). Corresponding point: $(Pd)")
    # @printf("%5d | %10.6f\n", n_evals, fd)
    eval_data[n_evals] = fd
    return (true, true, [fd])
end

function compute_subproblem_bounds(M::AbstractManifold, p)
    q = manifold_dimension(M)
    return -10.0 .* ones(q), 10.0 .* ones(q)
end

function rDFO(M::AbstractManifold, f, p0; kwargs...)
    mco = ManifoldCostObjective(f)
    return rDFO(M, mco, p0; kwargs...)
end

function rDFO(
        M::AbstractManifold,
        mco::AbstractManifoldCostObjective,
        p0,
        em::AbstractEvalManager;
        stopping_criterion::DFStoppingCriterion = StopWhenWithinRadius(),
        retraction_method::AbstractRetractionMethod = default_retraction_method(M)
    )
    rDFOs = RDFOState(M, p0, stopping_criterion, retraction_method)
    q = manifold_dimension(M)
    iter::Int = 0
    p = p0
    global n_evals
    n_evals = 0

    while !stopping_criterion(M, rDFOs, iter, n_evals)
        # Construct the local blackbox within ℝ^q
        local_bb(v) = local_bb_wrapper(mco, M, p, v)

        # Create the q-dimensional problem for NOMAD
        lb, ub = compute_subproblem_bounds(M, p)
        local_problem_budget = get_eval_budget(em)
        local_problem_options = NOMAD.NomadOptions(max_bb_eval = local_problem_budget, display_degree = 0)
        local_problem = NomadProblem(q, 1, ["OBJ"], local_bb; lower_bound = lb, upper_bound = ub, options = local_problem_options)

        # Solve the subproblem
        local_result = solve(local_problem, zeros(q))
        v = local_result.x_best_feas
        d = get_vector(M, p, v)
        new_p = retract(M, p, d)
        best_f = f(new_p)
        println("f(p) = $(best_f), n_evals=$(n_evals)")
        p = new_p

        iter += 1 # TODO. Replace the affectation part with modifications on the solver state.
    end
    return
end

function rmads(M::AbstractManifold, f, max_bb_eval::Int; p0 = rand(M), kwargs...)
    q = manifold_dimension(M)
    ℓ = 0
    # Instantiating a solver state, or anything that stores at least the stopping criterion.
    p = p0

    global n_evals, eval_data
    n_evals = 0
    eval_data = fill(typemax(Float64), max_bb_eval)

    while n_evals < max_bb_eval
        # Construct the local blackbox within ℝ^q
        n_evals == 0 && println(p)
        local_bb(v) = local_bb_wrapper(f, M, p, v)

        # Create the q-dimensional problem for NOMAD
        lb, ub = compute_subproblem_bounds(M, p)
        local_problem_budget = div(max_bb_eval, 10)
        local_problem_options = NOMAD.NomadOptions(max_bb_eval = local_problem_budget, display_degree = 0)
        local_problem = NomadProblem(q, 1, ["OBJ"], local_bb; lower_bound = lb, upper_bound = ub, options = local_problem_options)

        # Solve the subproblem
        local_result = solve(local_problem, zeros(q))
        vℓ = local_result.x_best_feas
        dℓ = get_vector(M, p, vℓ)
        new_p = retract(M, p, dℓ)
        fℓ = f(new_p)
        # println("f(p) = $(fℓ), n_evals=$(n_evals)")
        η = norm(p .- new_p)
        p = new_p

        ℓ += 1
    end
    return p, eval_data
end

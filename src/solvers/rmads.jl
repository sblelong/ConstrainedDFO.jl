using ManifoldsBase
using NOMAD

export rmads

function local_bb_wrapper(f, M::AbstractManifold, p, v)
    d = get_vector(M, p, v)
    Pd = retract(M, p, d, ProjectionRetraction())
    return (true, true, [f(Pd)])
end

function compute_subproblem_bounds(M::AbstractManifold, p)
    q = manifold_dimension(M)
    return -10.0 .* ones(q), 10.0 .* ones(q)
end

function rmads(M::AbstractManifold, f, max_bb_eval::Int, p0 = rand(M); kwargs...)
    q = manifold_dimension(M)
    ℓ = 0
    # Instantiating a solver state, or anything that stores at least the stopping criterion.
    p = p0

    while ℓ < 100
        # Construct the local blackbox within ℝ^q
        local_bb(v) = local_bb_wrapper(f, M, p, v)

        # Create the q-dimensional problem for NOMAD
        lb, ub = compute_subproblem_bounds(M, p)
        local_problem_budget = div(max_bb_eval, 10)
        local_problem_options = NOMAD.NomadOptions(max_bb_eval = local_problem_budget)
        local_problem = NomadProblem(q, 1, ["OBJ"], local_bb; lower_bound = lb, upper_bound = ub, options = local_problem_options)

        # Solve the subproblem
        local_result = solve(local_problem, zeros(q))
        vℓ = local_result.x_best_feas
        dℓ = get_vector(M, p, vℓ)
        new_p = retract(M, p, dℓ)
        fℓ = f(new_p)
        println("f(p) = $(fℓ)")
        η = norm(p .- new_p)
        p = new_p

        ℓ += 1
    end
    return p
end

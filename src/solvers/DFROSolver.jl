function DFROSolver(BI::BlackboxInstance; kwargs...)
    M = EqualityManifold(BI.problem)
    f(p) = eval_objective(BI, p)
    g(p) = eval_ineqs(BI, p)

    return DFROSolver(M, f, g, get_x0(BI), get_n_ineqs(BI); kwargs...)
end

"""
    DFROSolver(M::AbstractManifold, f, g, p0, n_ineqs::Int; kwargs...)
    DFROSolver(BI::BlackboxInstance; kwargs...)

# Arguments
- `M` is the Riemannian submanifold of ``\\mathbb{R}^n`` where the optimization problem is solved.
- `f` is the cost function.
- `g` gives the inequality constraints.
- `p0` is the starting point used by the solver. If ``p_0\\notin\\mathcal{M}``, then the actual starting point used is ``\\mathrm{proj}_{\\mathcal{M}}(p_0)``.
- `n_ineqs` is the number of inequality constraints.

A sole argument can also be given as a [`BlackboxInstance`](@ref).

# Keyword arguments
Keyword arguments can include:
- `tangent_solver::AbstractTangentSolver` is the tangent solver used. Defaults to [`MADSTangentSolver`](@ref).
- `max_evals::Int` is the budget of evaluations to solve the problem. Defaults to ``1000\\times n`` where ``n`` is the dimension of the problem.
- `retraction_method::AbstractRetractionMethod` is the method used to retract tangent vectors to the manifold while solving subproblems. Defaults to `default_retraction_method(M)`.
- `invertibility_bound::AbstractInvertibilityBound` is the formula used to compute a lower bound on the invertibility radius of ``\\mathcal{M}``. Defaults to `default_invertibility_bound(M)` (see [`invertibility_radius`](@ref) and [`default_invertibility_bound`](@ref)).
- `tol_eqs::Float64` is the numerical tolerance below which a point is considered feasible for equality constraints. Defaults to ``1.0\\times 10^{-8}``.
- `tol_ineqs::Float64` is the numerical tolerance below which a point is considered feasible for inequality constraints. Defaults to ``1.0\\times 10^{-8}``.
"""
function DFROSolver(M::AbstractManifold, f, g, p0, n_ineqs::Int; kwargs...)
    cost(M, p) = f(p)
    mco = ManifoldCostObjective(cost)
    return DFROSolver(M, mco, g, p0, n_ineqs; kwargs...)
end

function DFROSolver(
        M::AbstractManifold,
        mco::AbstractManifoldCostObjective,
        g,
        p0,
        m::Int;
        tangent_solver::AbstractTangentSolver = MADSTangentSolver(),
        max_evals::Int = 1000 * representation_size(M)[1],
        retraction_method::AbstractRetractionMethod = default_retraction_method(M),
        invertibility_bound::AbstractInvertibilityBound = default_invertibility_bound(M),
        tol_eqs::Float64 = 1.0e-8,
        tol_ineqs::Float64 = 1.0e-8,
        print_level::Int = 1,
        display_first_infeasible::Bool = true
    )

    manifold_dimension(M) ≤ 0 && throw(NumericalError("ConstrainedDFO.jl error: calling DFROSolver with a manifold with dimension < 1."))

    n_eqs = representation_size(M)[1] - manifold_dimension(M) # Basic assumption: M is a (n-p)-dimensional manifold.

    if print_level == 1
        header = @sprintf(
            "Solving with DFRO solver.\n Problem size: %i\n Size of tangent spaces: %i",
            representation_size(M)[1], manifold_dimension(M)
        )
        println(header)
    end

    if is_point_dispatcher(M, p0; tol_eqs = tol_eqs)
        p = p0
    else
        if display_first_infeasible
            fp0 = get_cost(M, mco, p0)
            hp0 = abs.(eval_defining_function(M, p0))
            gp0 = g(p0)
            extra_line = @sprintf(
                "%-10s%-20.10g",
                0, fp0
            ) * join((@sprintf("%-20.10g", hp0[i]) for i in 1:n_eqs)) * join((@sprintf("%-20.10g", gp0[i]) for i in 1:m))
            println(extra_line)
        end

        p = project(M, p0)
    end

    outer_counter = 0
    p = is_point_dispatcher(M, p0; tol_eqs = tol_eqs) ? p0 : project(M, p0)
    remaining_eval_budget = max_evals
    termination::Bool = false
    while !termination
        outer_counter += 1

        # Compute a lower bound to the invertibility radius at p
        radius = invertibility_radius(M, p; m = retraction_method, ρ = invertibility_bound)

        if print_level == 1
            first_line_log = @sprintf(
                "Starting outer iteration #%s. Invertibility radius used at current solution: %.6f",
                outer_counter, radius
            )
            println(first_line_log)
            header_log = @sprintf(
                "%-10s%-20s",
                "eval", "objective"
            ) * join((@sprintf("%-20s", "h") for _ in 1:n_eqs)) * join((@sprintf("%-20s", "g") for _ in 1:m))
            println(header_log)
        end

        # Solve the subproblem in the current tangent space
        solve!(tangent_solver, mco, M, p, retraction_method, radius, m; max_evals = remaining_eval_budget, εeqs = tol_eqs)

        # Retrieve data from the tangent solver
        data_f = get_data_f(tangent_solver)
        data_Rpv = get_data_Rpv(tangent_solver)
        data_g = get_data_g(tangent_solver)
        data_h = get_data_h(tangent_solver)
        n_evals = length(data_f)
        radius_evaluation = get_radius_evaluation(tangent_solver)
        improvement_outside_radius = radius_evaluation > 0
        last_eval = improvement_outside_radius ? radius_evaluation : n_evals

        # Print data from the tangent solver
        if print_level == 1
            first_eval = outer_counter == 1 ? 1 : 2 # Since x_{ℓ-1}^last and x_ℓ^1 are the same, we don't count the evaluation twice.
            for (number, eval) in enumerate(first_eval:last_eval)
                line_log = @sprintf(
                    "%-10s%-20.10g",
                    number, data_f[eval]
                ) * join((@sprintf("%-20.10g", data_h[eval][i]) for i in 1:n_eqs)) * join((@sprintf("%-20.10g", data_g[eval][i]) for i in 1:m))
                println(line_log)
            end
        end

        # Find the solution of the subproblem in the logs and make it the new iterate
        # Be careful: do not look for the best value of f amongst the points that were only virtually evaluated.
        best_evaluation = argmin(data_f[1:last_eval])
        p = data_Rpv[best_evaluation]

        # Update remaining evaluations
        remaining_eval_budget -= last_eval

        termination = get_radius_evaluation(tangent_solver) == 0 || remaining_eval_budget == 0

        (print_level == 1 && get_radius_evaluation(tangent_solver) > 0) && println("Improvement was found outside the invertibility region. Switching to a new subproblem.\n")
    end

    end_message = remaining_eval_budget == 0 ? "EXIT: Maximum amount of blackbox evaluations used." : "EXIT: Subproblem solved within invertibility region."
    print_level == 1 && println(end_message)

    return p
end

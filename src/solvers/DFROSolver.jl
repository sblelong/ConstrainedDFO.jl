"""
    DFROSolver(M::AbstractManifold, f, g, p0, n_ineqs::Int; kwargs...)

# Arguments
- `M` is the Riemannian submanifold of ``\\mathbb{R}^n`` where the optimization problem is solved.
- `f` is the cost function.
- `g` gives the inequality constraints.
- `p0` is the starting point used by the solver. If ``p_0\\notin\\mathcal{M}``, then the actual starting point used is ``\\mathrm{proj}_{\\mathcal{M}}(p_0)``.
- `n_ineqs` is the number of inequality constraints.

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
        print_level::Int = 1
    )

    if print_level == 1
        separator = @sprintf(
            "%s%s%s%s%s",
            "-"^11, "-"^12, "-"^10, "-"^15, "-"^11
        )
        header = @sprintf(
            " %-10s%-12s%-10s%-15s%-10s",
            "Outer", "Radius", "Inner", "Objective", "‖v‖≥ρ"
        )

        println(separator)
        println(header)
        println(separator)
    end

    outer_counter = 0
    p = p0
    remaining_eval_budget = max_evals
    termination::Bool = false
    while !termination
        outer_counter += 1

        # Compute a lower bound to the invertibility radius at p
        radius = invertibility_radius(M, p; m = retraction_method, ρ = invertibility_bound)

        # Solve the subproblem in the current tangent space
        solve!(tangent_solver, mco, M, p, retraction_method, radius, m; max_evals = remaining_eval_budget, εeqs = tol_eqs)

        # Retrieve data from the tangent solver
        data_f = get_data_f(tangent_solver)
        data_Rpv = get_data_Rpv(tangent_solver)
        data_d = get_data_d(tangent_solver)
        n_evals = length(data_f)
        radius_evaluation = get_radius_evaluation(tangent_solver)
        improvement_outside_radius = radius_evaluation > 0

        # Print data from the tangent solver
        # First line: display outer_counter and ρ
        if print_level == 1
            first_line_log = @sprintf(
                " %-10d%-12.6f%-10d%-15.6f%-10s%-20s%-10s",
                outer_counter, radius, 1, data_f[1], "", data_Rpv[1], is_point(M, data_Rpv[1])
            )
            println(first_line_log)
        end
        # Then, display the rest
        last_eval = improvement_outside_radius ? radius_evaluation : n_evals
        if print_level == 1
            for eval in 2:(last_eval - 1)
                line_log = @sprintf(
                    " %-10s%-12s%-10d%-15.6f%-10s%-20s%-10s",
                    "", "", eval, data_f[eval], "", data_Rpv[eval], is_point(M, data_Rpv[eval])
                )
                println(line_log)
            end
            last_line_log = @sprintf(
                " %-10s%-12s%-10d%-15.6f%-10s%-20s%-10s",
                "", "", last_eval, data_f[last_eval], improvement_outside_radius ? "✓" : "✗", data_Rpv[last_eval], is_point(M, data_Rpv[last_eval])
            )
            println(last_line_log)
        end

        # Find the solution of the subproblem in the logs and make it the new iterate
        # Be careful: do not look for the best value of f amongst the points that were only virtually evaluated.
        best_evaluation = argmin(data_f[1:last_eval])
        p = data_Rpv[best_evaluation]

        # Update remaining evaluations
        remaining_eval_budget -= last_eval

        termination = get_radius_evaluation(tangent_solver) == 0 || remaining_eval_budget == 0
    end
    return p
end

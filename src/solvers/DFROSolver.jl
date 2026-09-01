function DFROSolver(M::AbstractManifold, f, g, p0, n_ineqs::Int; kwargs...)
    cost(M, p) = f(p)
    mco = ManifoldCostObjective(cost)
    return DFROSolver(M, mco, g, p0, n_ineqs; kwargs...)
end

"""
In the top-level function
- build the DFROState object
- manage whether to return the DFROState object
- possibly, redirect stdout to a stat file (that does not necessarily meet the format of the RunnerPost: a converter should be implemented later).
- project p0 to M if it is not feasible from the beginning.
"""
function DFROSolver(
        M::AbstractManifold,
        mco::AbstractManifoldCostObjective,
        g,
        p0,
        m::Int,
        tangent_solver::AbstractTangentSolver = MADSTangentSolver(),
        max_evals::Int = 1000 * representation_size(M)[1],
        retraction_method::AbstractRetractionMethod = default_retraction_method(M),
        invertibility_bound::AbstractInvertibilityBound = default_invertibility_bound(M),
        tol_eqs::Float64 = 1.0e-8,
        tol_ineqs::Float64 = 1.0e-8,
    )

    separator = @sprintf(
        "%s%s%s%s%s",
        "-"^4, "-"^6, "-"^4, "-"^11, "-"^5
    )
    header = @sprintf(
        " %-5s%-7s%-5s%-12s%-5s",
        "ℓ", "ρ", "k", "f", "‖v‖≥ρ"
    )

    println(header)
    println(separator)

    ℓ = 0
    p = p0
    remaining_eval_budget = max_evals
    termination::Bool = false
    while !termination
        ℓ += 1

        # Compute a lower bound to the invertibility radius at p
        radius = invertibility_radius(M, p; m = retraction_method, ρ = invertibility_bound)

        # Solve the subproblem in the current tangent space
        solve!(tangent_solver, mco, M, p, retraction_method, radius, m; max_evals = remaining_eval_budget, εeqs = tol_eqs)

        # Retrieve data from the tangent solver
        data_f = get_data_f(tangent_solver)
        data_Rpv = get_data_Rpv(tangent_solver)
        n_evals = length(data_f)
        radius_evaluation = tangent_solver.radius_evaluation
        solved_outside_radius = radius_evaluation > 0

        # Print data from the tangent solver
        # First line: display ℓ and ρ
        first_line_log = @sprintf(
            " %5d%7.3f%5d%12.6f%5s",
            ℓ, radius, 1, data_f[1], ""
        )
        println(first_line_log)
        # Then, display the rest
        last_eval = solved_outside_radius ? radius_evaluation : n_evals
        for eval in 2:(last_eval - 1)
            line_log = @sprintf(
                " %5s%7s%5d%12.6f%5s",
                "", "", eval, data_f[eval], ""
            )
            println(line_log)
        end
        last_line_log = @sprintf(
            " %5s%7s%5d%12.6f%5s",
            "", "", last_eval, data_f[last_eval], solved_outside_radius ? "✓" : "✗"
        )
        println(last_line_log)

        # Find the solution of the subproblem in the logs and make it the new iterate
        best_evaluation = argmin(data_f)
        p = data_Rpv[best_evaluation]

        # Update remaining evaluations
        remaining_eval_budget -= last_eval

        termination = get_radius_evaluation(tangent_solver) == 0 || remaining_eval_budget == 0
    end
    return p
end

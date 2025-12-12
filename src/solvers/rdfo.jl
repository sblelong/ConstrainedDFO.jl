using ManifoldsBase
using Manopt

export rDFO

function retract_eval(M::AbstractManifold, mco::AbstractManifoldCostObjective, p, v, retraction_method::AbstractRetractionMethod)
    d = get_vector(M, p, v, DefaultOrthonormalBasis())
    Pd = retract(M, p, d, retraction_method)
    fd = get_cost(M, mco, Pd)
    return (true, true, [fd])
end

function rDFO(
        M::AbstractManifold,
        f::Function,
        p0;
        solver::String = "mads",
        max_evals::Int = 1000 * representation_size(M)[1],
        stopping_criterion::DFStoppingCriterion = StopRadiusAndBudget(max_evals),
        retraction_method::AbstractRetractionMethod = default_retraction_method(M)
    )
    mco = ManifoldCostObjective(f)
    return rDFO(M, mco, p0; solver, stopping_criterion, retraction_method)
end

function rDFO(
        M::AbstractManifold,
        mco::AbstractManifoldCostObjective,
        p0;
        solver::String = "mads",
        max_evals::Int = 1000 * representation_size(M)[1],
        stopping_criterion::DFStoppingCriterion = StopRadiusAndBudget(max_evals),
        retraction_method::AbstractRetractionMethod = default_retraction_method(M)
    )
    rdfos = RDFOState(M, p0, stopping_criterion, retraction_method)
    mpb = DefaultManoptProblem(M, mco)

    q = manifold_dimension(M)
    iter::Int = 0
    n_evals::Int = 0
    eval_data::Vector{Float64} = fill(typemax(Float64), max_evals)
    p = p0
    remaining_evals = max_evals

    while true
        # Construct the subproblem blackbox in ℝ^q.
        local_blackbox(v) = retract_eval(M, mco, p, v, retraction_method) # Will embed v in TpM, then retract this embedding on M and finally evaluate the objective at this retracted point.

        # Compute the injectivity radius of M at p, or a lower bound.
        radius = injectivity_radius(M, p, retraction_method)

        # Solve the q-dimensional subproblem with the given solver, with stopping criterion being the injectivity radius (like a trust-region problem, except it just stops whenever the trust radius is reached, and the radius is never update. So, not really like a TR.). Don't forget to always include the maximum budget, even when it's not the main stopping criterion.

        # Solve the subproblem for budgets ranging arbitrarily from 10 to the max budget, in order to detect quickly if we're outside of the inversibility radius.
        subproblem_result = nothing
        eval_nb::Int = 0
        v = zeros(q)
        d = zeros(representation_size(M))
        flag = false
        for budget in (10, 50, remaining_evals)
            if budget > remaining_evals
                continue
            end
            subproblem_options = NOMAD.NomadOptions(max_bb_eval = budget, display_stats = ["BBE", "SOL", "OBJ"])
            subproblem = NomadProblem(q, 1, ["OBJ"], local_blackbox; options = subproblem_options)
            # Solve the subproblem and redirect the outputs to a .txt that will be read later.
            redirect_to_files("tmp.log", "tmp.err") do
                subproblem_result = solve(subproblem, zeros(q))
            end

            # Retrieve the solution to the subproblem as the first one whose norm exceeds the radius estimate. Update the amount of remaining evaluations accordingly.
            open("tmp.log", "r") do logf
                for line in eachline(logf)
                    if startswith(line, "!!")
                        println("Numerical error, skipping to the next point.")
                        flag = true
                        break
                    end
                    if occursin(r"^\d+", line)
                        numbers = [parse(Float64, m.match) for m in eachmatch(r"-?\d+\.?\d*", line)]
                        eval_nb = numbers[1]
                        v = numbers[2:(1 + q)]
                        d = get_vector(M, p, v)
                        eval_data[(n_evals + eval_nb):end] .= numbers[end]
                        if norm(d) ≥ radius
                            println("Radius reached at eval ", eval_nb)
                            flag = true
                            break
                        end
                    end
                end
            end
        end

        # Retract the solution found at previous step.
        d = get_vector(M, p, v)
        new_p = retract(M, p, d)
        best_f = get_cost(M, mco, new_p)
        p = new_p
        println(new_p, best_f)

        # Updates.
        iter += 1
        n_evals += eval_nb
        remaining_evals = max_evals - n_evals

        stopping_criterion(mpb, rdfos, iter, n_evals, flag) && break
    end
    println("End of optimization. Total evals: $(n_evals).")
    return p, eval_data
end

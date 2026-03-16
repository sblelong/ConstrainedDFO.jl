using ManifoldsBase
using Manopt
using Printf

export rDFO

"""
Retract the point ``v\\in T_p\\mathcal{M}`` and then evaluate the objective at this retracted point.
"""
retract_eval(M::AbstractManifold, mco::AbstractManifoldCostObjective, p, v, retraction_method::AbstractRetractionMethod, solver::AbstractDFRSolver)

function retract_eval(M::AbstractManifold, mco::AbstractManifoldCostObjective, p, v, retraction_method::AbstractRetractionMethod, solver::MADSDFRSolver)
    d = get_vector(M, p, v, DefaultOrthonormalBasis())
    Pd = retract(M, p, d, retraction_method)
    fd = get_cost(M, mco, Pd)
    return (true, true, [fd])
end

function rDFO(
        M::AbstractManifold,
        f::Function,
        p0;
        solver::AbstractDFRSolver = MADSDFRSolver(),
        max_evals::Int = 1000 * representation_size(M)[1],
        stopping_criterion::DFStoppingCriterion = StopRadiusAndBudget(max_evals),
        retraction_method::AbstractRetractionMethod = default_retraction_method(M),
        invertibility_bound::AbstractInvertibilityBound = default_invertibility_bound(M, retraction_method)
    )
    mco = ManifoldCostObjective(f)
    return rDFO(M, mco, p0; solver = solver, max_evals = max_evals, stopping_criterion = stopping_criterion, retraction_method = retraction_method, invertibility_bound = invertibility_bound)
end

function rDFO(
        M::AbstractManifold,
        mco::AbstractManifoldCostObjective,
        p0;
        solver::AbstractDFRSolver = MADSDFRSolver(),
        max_evals::Int = 1000 * representation_size(M)[1],
        stopping_criterion::DFStoppingCriterion = StopRadiusAndBudget(max_evals),
        retraction_method::AbstractRetractionMethod = default_retraction_method(M),
        invertibility_bound::AbstractInvertibilityBound = default_invertibility_bound(M, retraction_method)
    )
    rdfos = RDFOState(M, p0, stopping_criterion, retraction_method)
    mpb = DefaultManoptProblem(M, mco)

    q = manifold_dimension(M)
    iter::Int = 0
    n_evals::Int = 0
    eval_data::Vector{Float64} = fill(typemax(Float64), max_evals)
    remaining_evals = max_evals
    processed_solver_details = Dict()

    p = is_point(M, p0) ? p0 : project(M, p0)

    if typeof(solver) == MADSDFRSolver
        options = Dict()
        if solver.transfer_mesh_size
            processed_solver_details = Dict("best_mesh_size" => ones(q))
        end
    end

    iterates_history = Matrix{Float64}[]
    objective_history = Vector{Float64}[]

    v_history = Matrix{Float64}[]
    d_history = Matrix{Float64}[]

    # @printf("| %10s | %10s | %10s | %13s |\n", "iteration", "# evals", "total evals", "f")
    # @printf("|-%10s-|-%10s-|-%11s-|-%13s-|\n", "-"^10, "-"^10, "-"^10, "-"^14)


    while true
        # Construct the subproblem blackbox in ℝ^q.
        local_blackbox(v) = retract_eval(M, mco, p, v, retraction_method, solver) # Will embed v in TpM, then retract this embedding on M and finally evaluate the objective at this retracted point.

        # Compute the injectivity radius, or a lower bound to it.
        radius = invertibility_radius(M, p, retraction_method, invertibility_bound)

        # Solve the q-dimensional subproblem with the given solver, with stopping criterion being the injectivity radius.

        if typeof(solver) == MADSDFRSolver && solver.transfer_mesh_size
            options["initial_mesh_size"] = processed_solver_details["best_mesh_size"]
        end

        solve!(M, p, solver, local_blackbox, radius, remaining_evals; options)

        vs, fs, solver_details = get_subproblem_result(M, solver)
        # TODO. Check if instead of embedding and evaluating again, the evaluation wrappers (defined in this file) could use global variables to store the data.

        # Embed the vs in ℝ^n
        ds = vs_to_ds(M, p, vs)

        # Retract the ds on M
        Rpds = ds_to_Rpds(M, p, ds)

        # Retrieve the costs of all evaluated points
        fs = vectorized_cost(M, mco, Rpds)
        stratified_fs = fill(typemax(Float64), length(fs))
        best = fs[1]
        for i in 1:length(stratified_fs)
            if fs[i] < best
                best = fs[i]
            end
            stratified_fs[i] = best
        end

        # Retrieve the best point
        best_f, best_eval = findmin(fs)
        best_v, best_d, best_p = vs[best_eval, :], ds[best_eval, :], Rpds[best_eval, :]

        # Check how many evaluations were actually used
        used_evals = size(vs)[1]

        # Updates
        set_tangent_iterate!(rdfos, best_d)
        set_iterate!(rdfos, best_p)
        eval_data[(n_evals + 1):(n_evals + used_evals)] .= stratified_fs

        p = best_p
        iter += 1
        n_evals += used_evals
        remaining_evals -= used_evals

        push!(iterates_history, Rpds)
        push!(objective_history, fs)
        push!(v_history, vs)
        push!(d_history, ds)

        processed_solver_details = process_details(M, solver, solver_details)

        # @printf("| %10d | %10d | %11d | %14e |\n", iter, used_evals, n_evals, best_f)

        stopping_criterion(mpb, rdfos, iter, n_evals, solver.flag, retraction_method, invertibility_bound) && break
    end
    # println("A stopping criterion was met.")
    return p, eval_data[1:n_evals], iterates_history, objective_history, v_history, d_history
end

function vs_to_ds(M::AbstractManifold, p, vs)
    n_evals = size(vs)[1]
    q = representation_size(M)[1]
    ds = zeros(Float64, (n_evals, q))
    n_evals = size(ds)[1]
    for i in 1:n_evals
        v = vs[i, :]
        d = get_vector(M, p, v)
        ds[i, :] .= d
    end
    return ds
end

function ds_to_Rpds(M::AbstractManifold, p, ds; retraction_method::AbstractRetractionMethod = default_retraction_method(M))
    Rpds = zeros(Float64, size(ds))
    n_evals = size(ds)[1]
    for i in 1:n_evals
        d = ds[i, :]
        Rpd = retract(M, p, d, retraction_method)
        Rpds[i, :] .= Rpd
    end
    return Rpds
end

function vectorized_cost(M::AbstractManifold, mco::AbstractManifoldCostObjective, Rpds)
    n_evals = size(Rpds)[1]
    fs = zeros(n_evals)
    for i in 1:n_evals
        f = get_cost(M, mco, Rpds[i, :])
        fs[i] = f
    end
    return fs
end

"""
Retract the point ``v\\in T_p\\mathcal{M}`` and then evaluate the objective at this retracted point.
"""
retract_eval(M::AbstractManifold, mco::AbstractManifoldCostObjective, p, v, retraction_method::AbstractRetractionMethod, solver::AbstractTangentSolver)

function retract_eval(M::AbstractManifold, mco::AbstractManifoldCostObjective, p, v, retraction_method::AbstractRetractionMethod, solver::MADSDFRSolver; inequality_constraints::Union{Function, Nothing} = nothing, nb_inequalities::Int = 0, εeqs::Float64 = 1.0e-8)
    return try
        d = get_vector(M, p, v, DefaultOrthonormalBasis())
        Pd = retract(M, p, d, retraction_method)
        fd = is_point(M, Pd; atol = εeqs) ? [get_cost(M, mco, Pd)] : [1.0e20]
        if isnothing(inequality_constraints)
            return (true, true, fd)
        else
            gd = inequality_constraints(Pd)
            return (true, true, [fd; gd])
        end
    catch e
        println("FAAAILED: $(e)")
        if isnothing(inequality_constraints)
            return (true, true, [1.0e20])
        else
            return (true, true, [[1.0e20] ; [1.0e20 for _ in 1:nb_inequalities]])
        end
    end
end

"""
    TODO.
"""
function DFROSolver(
        M::AbstractManifold,
        f::Function,
        p0;
        inequality_constraints::Union{Function, Nothing} = nothing,
        solver::AbstractTangentSolver = MADSDFRSolver(),
        max_evals::Int = 1000 * representation_size(M)[1],
        stopping_criterion::DFStoppingCriterion = StopRadiusAndBudget(max_evals),
        retraction_method::AbstractRetractionMethod = default_retraction_method(M),
        invertibility_bound::AbstractInvertibilityBound = default_invertibility_bound(M, retraction_method),
        εeqs::Float64 = 1.0e-8
    )
    mco = ManifoldCostObjective(f)
    return DFRO(M, mco, p0; inequality_constraints = inequality_constraints, solver = solver, max_evals = max_evals, stopping_criterion = stopping_criterion, retraction_method = retraction_method, invertibility_bound = invertibility_bound, εeqs = εeqs)
end

function DFROSolver(
        M::AbstractManifold,
        mco::AbstractManifoldCostObjective,
        p0;
        inequality_constraints::Union{Function, Nothing} = nothing,
        solver::AbstractTangentSolver = MADSDFRSolver(),
        max_evals::Int = 1000 * representation_size(M)[1],
        stopping_criterion::DFStoppingCriterion = StopRadiusAndBudget(max_evals),
        retraction_method::AbstractRetractionMethod = default_retraction_method(M),
        invertibility_bound::AbstractInvertibilityBound = default_invertibility_bound(M, retraction_method),
        εeqs::Float64 = 1.0e-8
    )
    rdfos = DFROState(M, p0, stopping_criterion, retraction_method)
    mpb = DefaultManoptProblem(M, mco)

    n = representation_size(M)[1]
    q = manifold_dimension(M)
    iter::Int = 0
    n_evals::Int = 0
    eval_data::Vector{Float64} = fill(typemax(Float64), max_evals)
    remaining_evals = max_evals
    processed_solver_details = Dict()

    p = is_point(M, p0; atol = εeqs) ? p0 : project(M, p0)
    !is_point(M, p; atol = εeqs) && return p, [1.0e20], [], [], [], [], []

    if typeof(solver) == MADSDFRSolver
        options = Dict()
        if solver.transfer_mesh_size
            processed_solver_details = Dict("best_mesh_size" => ones(q))
        end
    end

    iterates_history = Matrix{Float64}[]
    objective_history = Vector{Float64}[]
    if !isnothing(inequality_constraints)
        g_history = Matrix{Float64}[]
    end

    v_history = Matrix{Float64}[]
    d_history = Matrix{Float64}[]
    main_iterates = Vector{Float64}[]

    # @printf("| %10s | %10s | %10s | %13s |\n", "iteration", "# evals", "total evals", "f")
    # @printf("|-%10s-|-%10s-|-%11s-|-%13s-|\n", "-"^10, "-"^10, "-"^10, "-"^14)


    while true
        push!(main_iterates, p)
        # Construct the subproblem blackbox in ℝ^q.
        local_blackbox(v) = retract_eval(M, mco, p, v, retraction_method, solver; inequality_constraints = inequality_constraints, εeqs = εeqs) # Will embed v in TpM, then retract this embedding on M and finally evaluate the objective at this retracted point.

        # Compute the injectivity radius, or a lower bound to it.
        radius = invertibility_radius(M, p, retraction_method, invertibility_bound)

        # Solve the q-dimensional subproblem with the given solver, with stopping criterion being the injectivity radius.

        if typeof(solver) == MADSDFRSolver && solver.transfer_mesh_size
            options["initial_mesh_size"] = processed_solver_details["best_mesh_size"]
        end

        try
            if isnothing(inequality_constraints)
                solve!(M, p, solver, local_blackbox, radius, remaining_evals; options)
            else
                gp = inequality_constraints(p)
                nb_inequalities = length(gp)
                init_feasible = all(gp .≤ 0)
                solve!(M, p, solver, local_blackbox, radius, remaining_evals; options, nb_inequalities = nb_inequalities, init_feasible = init_feasible)
            end
        catch e
            # An error thrown here means that solving the subproblem has failed.
            # It is most likely due to a Jacobian that does not have full rank at the current iterate.
            eval_data[1] = get_cost(M, mco, p)
            return p, eval_data[1:1], iterates_history, objective_history, v_history, d_history
        end

        if isnothing(inequality_constraints)
            vs, fs, solver_details = get_subproblem_result(M, solver)
        else
            gp = inequality_constraints(p)
            nb_inequalities = length(gp)
            vs, fs, gs, solver_details = get_subproblem_result(M, solver; nb_inequalities = nb_inequalities)
        end
        # TODO. Check if instead of embedding and evaluating again, the evaluation wrappers (defined in this file) could use global variables to store the data.

        # Embed the vs in ℝ^n
        ds = vs_to_ds(M, p, vs)

        # Retract the ds on M
        Rpds = ds_to_Rpds(M, p, ds)

        # Retrieve the costs of all evaluated points
        fs = vectorized_cost(M, mco, Rpds)
        stratified_fs = fill(typemax(Float64), length(fs))

        if isnothing(inequality_constraints)
            best = fs[1]
            for i in 1:length(stratified_fs)
                if fs[i] < best
                    best = fs[i]
                end
                stratified_fs[i] = best
            end
        else
            # Retrieve the values of the inequality constraints for all evaluated points
            gp = inequality_constraints(p)
            nb_inequalities = length(gp)
            gs = vectorized_inequalities(inequality_constraints, Rpds, nb_inequalities)
            feasible_iterates = findall(i -> all(@inbounds(gs[i, j] ≤ 1.0e-8 for j in axes(gs, 2))), axes(gs, 1))
            feasible_fs = fs[feasible_iterates]
            best = 1.0e20

            for i in 1:length(stratified_fs)
                fval = (i in feasible_iterates) ? fs[i] : 1.0e20
                if fval < best
                    best = fval
                end
                stratified_fs[i] = best
            end
        end

        best_f::Float64 = 1.0e20
        best_eval::Int = 0
        best_v = zeros(q)
        best_d = zeros(n)
        best_p = zeros(n)

        # Retrieve the best point
        if isnothing(inequality_constraints)
            best_f, best_eval = findmin(fs)
            best_v, best_d, best_p = vs[best_eval, :], ds[best_eval, :], Rpds[best_eval, :]
        else
            feasible_iterates = findall(i -> all(@inbounds(gs[i, j] ≤ 1.0e-8 for j in axes(gs, 2))), axes(gs, 1))
            feasible_fs = fs[feasible_iterates]
            if length(feasible_fs) == 0
                best_f = 1.0e20
                best_eval = typemax(Int)
                best_v, best_d, best_p = fill(typemax(Float64), q), fill(typemax(Float64), n), fill(typemax(Float64), n)
            else
                best_f = minimum(feasible_fs)
                it::Int = 1
                while fs[it] ≠ best_f || gs[it] > 1.0e-8
                    it += 1
                end
                best_eval = it
                best_v, best_d, best_p = vs[best_eval, :], ds[best_eval, :], Rpds[best_eval, :]
            end
        end

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
        if !isnothing(inequality_constraints)
            push!(g_history, gs)
        end

        processed_solver_details = process_details(M, solver, solver_details)

        (norm(p) == typemax(Float64) || !is_point(M, p; atol = εeqs)) && break

        # @printf("| %10d | %10d | %11d | %14e |\n", iter, used_evals, n_evals, best_f)
        stopping_criterion(mpb, rdfos, iter, n_evals, solver.flag, retraction_method, invertibility_bound) && break
    end
    # println("A stopping criterion was met.")
    return p, eval_data[1:n_evals], iterates_history, objective_history, v_history, d_history, main_iterates
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

function vectorized_inequalities(g::Function, Rpds, nb_inequalities::Int)
    n_evals = size(Rpds)[1]
    gs = zeros((n_evals, nb_inequalities))
    for i in 1:n_evals
        g_val = g(Rpds[i, :])
        gs[i, :] = g_val
    end
    return gs
end

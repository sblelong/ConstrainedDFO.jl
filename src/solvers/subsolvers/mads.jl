using NOMAD
using ManifoldsBase

mutable struct MADSDFRSolver <: AbstractDFRSolver
    log_path::String
    last_eval::Int
    flag::Bool
    transfer_mesh_size::Bool
end

MADSDFRSolver() = MADSDFRSolver("./tmp.log", 0, false, true)
MADSDFRSolver(log_path::String) = MADSDFRSolver(log_path, 0, false, true)
MADSDFRSolver(transfer_mesh_size::Bool) = MADSDFRSolver("./tmp.log", 0, false, transfer_mesh_size)

set_log_path!(MS::MADSDFRSolver, s::String) = MS.log_path = s
set_last_eval!(MS::MADSDFRSolver, eval::Int) = MS.last_eval = eval
set_flag!(MS::MADSDFRSolver, val::Bool) = MS.flag = val

function solve!(M::AbstractManifold, p, MS::MADSDFRSolver, bb, radius, max_evals::Int; options::Dict = Dict(), nb_inequalities::Int = 0, init_feasible::Bool = false)
    q = manifold_dimension(M)
    # Check in a preventive way if a numerical stopping reason is reached before the budget is used
    for budget in (10, 50, max_evals)
        budget > max_evals && continue

        if nb_inequalities > 0
            noptions = NOMAD.NomadOptions(max_bb_eval = budget, display_stats = [["BBE", "SOL", "OBJ", "MESH_INDEX"] ; ["CONS_H" for _ in 1:nb_inequalities]], display_all_eval = true)
        else
            noptions = NOMAD.NomadOptions(max_bb_eval = budget, display_stats = ["BBE", "SOL", "OBJ", "MESH_INDEX"], display_all_eval = true)
        end

        initial_mesh_size = haskey(options, "initial_mesh_size") ? options["initial_mesh_size"] : ones(q)

        problem = NomadProblem(q, 1 + nb_inequalities, [["OBJ"] ; ["EB" for _ in 1:nb_inequalities]], bb; initial_mesh_size = initial_mesh_size, options = noptions)

        # Redirect the output to a file that will be processed after
        redirect_to_files(MS.log_path) do
            result = solve(problem, zeros(q))
        end

        # Process the output
        flag::Bool = false
        eval_nb::Int = 0
        best_f::Float64 = Inf
        if !init_feasible
            best_infeas::Vector{Float64} = fill(Inf, nb_inequalities)
        end
        open(MS.log_path, "r") do logf
            for line in eachline(logf)
                if startswith(line, "!!")
                    flag = true
                    set_last_eval!(MS, eval_nb)
                    break
                end
                if occursin(r"^\d+", line)
                    parts = split(line)
                    eval_nb = parse(Int, parts[1])
                    bracket_start = findfirst(isequal('('), line)
                    bracket_end = findfirst(isequal(')'), line)
                    bracket_content = line[(bracket_start + 1):(bracket_end - 1)]
                    v = parse.(Float64, split(bracket_content))
                    d = get_vector(M, p, v)
                    remaining = split(line[(bracket_end + 1):end])
                    f = parse(Float64, remaining[1])

                    # numbers = [parse(Float64, m.match) for m in eachmatch(r"-?\d+\.?\d*", line)]
                    # eval_nb = numbers[1]
                    # v = numbers[2:(1 + q)]
                    # d = get_vector(M, p, v)
                    # f = numbers[2 + q]

                    if nb_inequalities > 0
                        # If there are inequalities:
                        # 1. Retrieve the part of remaining where infeas_h is displayed by maybe getting rid of the (Phase One) part.
                        # 2. Parse this part of remaining with numbers and potentially "inf"
                        # 3. Turn this into a vector of Float64 (numbers and Inf).
                        remaining = remaining[2:(1 + nb_inequalities)]
                        infeas_h = [s == "inf" ? typemax(Float64) : parse(Float64, s) for s in remaining]
                        if f < best_f && all(infeas_h == 0.0)
                            best_f = f
                            if norm(d) ≥ radius
                                flag = true
                                set_last_eval!(MS, eval_nb)
                            end
                        end
                    elseif f < best_f
                        best_f = f
                        if norm(d) ≥ radius
                            flag = true
                            set_last_eval!(MS, eval_nb)
                            break
                        end
                    end
                end
            end
        end
        set_flag!(MS, flag)
        flag && break
        set_last_eval!(MS, eval_nb)
    end
    return MS
end

function get_subproblem_result(M::AbstractManifold, MS::MADSDFRSolver; nb_inequalities::Int = 0)
    q = manifold_dimension(M)

    vs = zeros((MS.last_eval, q))
    fs = zeros(MS.last_eval)
    mesh_indices = zeros((MS.last_eval, q))

    if nb_inequalities > 0
        gs = zeros((MS.last_eval, nb_inequalities))
    end

    return open(MS.log_path, "r") do logf
        if nb_inequalities > 0
            for line in eachline(logf)
                if occursin(r"^\d+", line)
                    parts = split(line)
                    n_eval = parse(Int, parts[1])
                    n_eval > MS.last_eval && break
                    bracket_start = findfirst(isequal('('), line)
                    bracket_end = findfirst(isequal(')'), line)
                    bracket_content = line[(bracket_start + 1):(bracket_end - 1)]
                    v = parse.(Float64, split(bracket_content))
                    remaining = split(line[(bracket_end + 1):end])
                    f = parse(Float64, remaining[1])
                    mesh_index = (n_eval == 1) ? ones(q) : parse.(Float64, remaining[2:(1 + q)])
                    if occursin("Phase One", line)
                        remaining = remaining[1:(end - 3)]
                    end
                    first_ineq_index = length(remaining) - nb_inequalities + 1
                    g = [s == "inf" ? typemax(Float64) : parse(Float64, s) for s in remaining[first_ineq_index:end]]
                    vs[n_eval, :] .= v
                    fs[n_eval] = f0
                    mesh_indices[n_eval, :] .= mesh_index
                    gs[n_eval, :] .= g
                end
            end
            details = Dict("mesh_indices" => mesh_indices)
            return vs, fs, gs, details
        else
            for line in eachline(logf)
                if occursin(r"^\d+", line)
                    parts = split(line)
                    n_eval = parse(Int, parts[1])
                    n_eval > MS.last_eval && break
                    bracket_start = findfirst(isequal('('), line)
                    bracket_end = findfirst(isequal(')'), line)
                    bracket_content = line[(bracket_start + 1):(bracket_end - 1)]
                    v = parse.(Float64, split(bracket_content))
                    remaining = split(line[(bracket_end + 1):end])
                    f = parse(Float64, remaining[1])
                    mesh_index = (n_eval == 1) ? ones(q) : parse.(Float64, remaining[2:(1 + q)])
                    vs[n_eval, :] .= v
                    fs[n_eval] = f
                    mesh_indices[n_eval, :] .= mesh_index
                end
            end
        end
        details = Dict("mesh_indices" => mesh_indices)
        return vs, fs, details
    end
end

function process_details(M::AbstractManifold, MS::MADSDFRSolver, details::Dict)
    processed_details = Dict()
    q = manifold_dimension(M)
    if haskey(details, "mesh_indices")
        best_mesh_index = details["mesh_indices"][MS.last_eval, :]
        best_mesh_size = ones(q) .* (1 / 2) .^ best_mesh_index
        processed_details["best_mesh_size"] = best_mesh_size
    end
    return processed_details
end

export MADSDFRSolver, solve!, process_details

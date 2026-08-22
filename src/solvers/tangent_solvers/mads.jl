"""
    AbstractNOMADBarrierType

Abstract type to choose between ways to handle inequality constraints with a MADS solver.
"""
abstract type AbstractNOMADBarrierType end

"""
    ExtremeBarrier <: AbstractNOMADBarrierType

Use the extreme barrier as a way to handle inequality constraints with a MADS solver.
"""
struct ExtremeBarrier <: AbstractNOMADBarrierType end

"""
    ProgressiveBarrier <: AbstractNOMADBarrierType

Use the progressive barrier as a way to handle inequality constraints with a MADS solver.
"""
struct ProgressiveBarrier <: AbstractNOMADBarrierType end

"""
    MADSTangentSolver <: AbstractTangentSolver

Subsolver using the Mesh Adaptive Direct Search (MADS) algorithm in tangent spaces. To be used together with the master solver [`DFROSolver`](@ref).

Note: the implementation makes use of the interface to the NOMAD 3 software offered by [`NOMAD.jl`](@extref NOMAD.solve)
"""
mutable struct MADSTangentSolver <: AbstractTangentSolver
    log_path::String
    flag::Bool
    barrier::AbstractNOMADBarrierType
    data_d::Vector{Vector{Float64}}
    data_Rpv::Vector{Vector{Float64}}
    data_f::Vector{Float64}
    data_g::Vector{Vector{Float64}}
end

MADSTangentSolver() = MADSTangentSolver("./tmp.log", false, ExtremeBarrier(), Vector{Float64}[], Vector{Float64}[], Float64[], Vector{Float64}[])
MADSTangentSolver(log_path::String) = MADSTangentSolver(log_path, false, ExtremeBarrier(), Vector{Float64}[], Float64[], Vector{Float64}[])

set_log_path!(MS::MADSTangentSolver, s::String) = MS.log_path = s
set_last_eval!(MS::MADSTangentSolver, eval::Int) = MS.last_eval = eval
set_flag!(MS::MADSTangentSolver, val::Bool) = MS.flag = val

function _build_nomad_problem(B::ExtremeBarrier, q::Int, n_ineqs::Int, bb, nomad_options::NOMAD.NomadOptions)
    problem = NOMAD.NomadProblem(q, 1 + n_ineqs, [["OBJ"] ; ["EB" for _ in 1:n_ineqs]], bb; options = nomad_options)
    return problem
end

function _build_nomad_problem(B::ProgressiveBarrier, q::Int, n_ineqs::Int, bb, nomad_options::NOMAD.NomadOptions)
    problem = NOMAD.NomadProblem(q, 1 + n_ineqs, [["OBJ"] ; ["PB" for _ in 1:n_ineqs]], bb; options = nomad_options)
    return problem
end

function format_eval_data(MTS::MADSTangentSolver, eval_data::BlackboxTangentData)
    return (true, true, [eval_data.f ; eval_data.g])
end

function solve!(
        MTS::MADSTangentSolver,
        mco::AbstractManifoldCostObjective,
        M::AbstractManifold,
        p,
        R::AbstractRetractionMethod,
        ρ::AbstractInvertibilityBound,
        n_ineqs::Int;
        g = nothing, max_evals::Int = 1000 * manifold_dimension(M), εeqs::Float64 = 1.0e-8
    )
    q = manifold_dimension(M)
    radius = invertibility_radius(M, p; R, ρ)

    for budget in (10, max_evals)
        budget > max_evals && continue

        # Set display format in the NOMAD history file
        if n_ineqs > 0
            nomad_options = NOMAD.NomadOptions(max_bb_eval = budget, display_stats = [["BBE", "SOL", "OBJ"] ; ["CONS_H" for _ in 1:nb_inequalities]], display_all_eval = true)
        else
            nomad_options = NOMAD.NomadOptions(max_bb_eval = budget, display_stats = ["BBE", "SOL", "OBJ"], display_all_eval = true)
        end

        # Build the blackbox
        bb(v) = blackbox_wrapper_store!(MTS, M, p, R, mco, n_ineqs, g, v; εeqs)

        problem = _build_nomad_problem(MTS.barrier, q, n_ineqs, bb, nomad_options)

        redirect_to_files(MTS.log_path) do # TODO. Is it even useful to redirect to external files if the tangent solver object stores everything?
            result = solve(problem, zeros(q))
        end

        # Check if an improving solution was found outside of the invertibility radius. If so, break and discard the remainder from the storage: these evaluations should not exist.

        n_evals = length(MTS.data_d)
        improving_outside_radius = false
        best_feasible_f = MTS.data_f[1]
        if n_ineqs > 0
            for id_eval in eachindex(MTS.data_d)
                if (MTS.data_f[id_eval] < best_feasible_f) && (all(MTS.data_g[id_eval] .≤ 0.0)) # Basic strategy: a solution is considered good enough to interrupt if it is feasible and f is improving.
                    best_feasible_f = MTS.data_f[id_eval]
                    if norm(MTS.data_d[id_eval]) ≥ radius
                        improving_outside_radius = true
                        break
                    end
                end
            end
        else
            for id_eval in eachindex(MTS.data_d)
                if MTS.data_f[id_eval] < best_feasible_f
                    best_feasible_f = MTS.data_f[id_eval]
                    if norm(MTS.data_d[id_eval]) ≥ radius
                        improving_outside_radius = true
                        break
                    end
                end
            end
        end
        clear_storage!(MTS) # Very important! The storage should be cleared before trying with another budget, to prevent duplicates.
        improving_outside_radius && break
    end
    return MTS
end

# function solve!(M::AbstractManifold, p, MS::MADSTangentSolver, bb, radius, max_evals::Int; options::Dict = Dict(), nb_inequalities::Int = 0, init_feasible::Bool = false)
#     q = manifold_dimension(M)
#     # Check in a preventive way if a numerical stopping reason is reached before the budget is used
#     for budget in (10, 50, max_evals)
#         budget > max_evals && continue

#         if nb_inequalities > 0
#             noptions = NOMAD.NomadOptions(max_bb_eval = budget, display_stats = [["BBE", "SOL", "OBJ", "MESH_INDEX"] ; ["CONS_H" for _ in 1:nb_inequalities]], display_all_eval = true)
#         else
#             noptions = NOMAD.NomadOptions(max_bb_eval = budget, display_stats = ["BBE", "SOL", "OBJ", "MESH_INDEX"], display_all_eval = true)
#         end

#         problem = NomadProblem(q, 1 + nb_inequalities, [["OBJ"] ; ["EB" for _ in 1:nb_inequalities]], bb; initial_mesh_size = initial_mesh_size, options = noptions)

#         # Redirect the output to a file that will be processed after
#         redirect_to_files(MS.log_path) do
#             result = solve(problem, zeros(q))
#         end

#         # Process the output
#         flag::Bool = false
#         eval_nb::Int = 0
#         best_f::Float64 = Inf
#         if !init_feasible
#             best_infeas::Vector{Float64} = fill(Inf, nb_inequalities)
#         end
#         open(MS.log_path, "r") do logf
#             for line in eachline(logf)
#                 if startswith(line, "!!")
#                     flag = true
#                     set_last_eval!(MS, eval_nb)
#                     break
#                 end
#                 if occursin(r"^\d+", line)
#                     parts = split(line)
#                     eval_nb = parse(Int, parts[1])
#                     bracket_start = findfirst(isequal('('), line)
#                     bracket_end = findfirst(isequal(')'), line)
#                     bracket_content = line[(bracket_start + 1):(bracket_end - 1)]
#                     v = parse.(Float64, split(bracket_content))
#                     d = get_vector(M, p, v)
#                     remaining = split(line[(bracket_end + 1):end])
#                     f = parse(Float64, remaining[1])

#                     if nb_inequalities > 0
#                         # If there are inequalities:
#                         # 1. Retrieve the part of remaining where infeas_h is displayed by maybe getting rid of the (Phase One) part.
#                         # 2. Parse this part of remaining with numbers and potentially "inf"
#                         # 3. Turn this into a vector of Float64 (numbers and Inf).
#                         remaining = remaining[2:(1 + nb_inequalities)]
#                         infeas_h = [s == "inf" ? typemax(Float64) : parse(Float64, s) for s in remaining]
#                         if f < best_f && all(infeas_h == 0.0)
#                             best_f = f
#                             if norm(d) ≥ radius
#                                 flag = true
#                                 set_last_eval!(MS, eval_nb)
#                             end
#                         end
#                     elseif f < best_f
#                         best_f = f
#                         if norm(d) ≥ radius
#                             flag = true
#                             set_last_eval!(MS, eval_nb)
#                             break
#                         end
#                     end
#                 end
#             end
#         end
#         set_flag!(MS, flag)
#         flag && break
#         set_last_eval!(MS, eval_nb)
#     end
#     return MS
# end

function get_subproblem_result(M::AbstractManifold, MS::MADSTangentSolver; nb_inequalities::Int = 0)
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
                    fs[n_eval] = f
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

function process_details(M::AbstractManifold, MS::MADSTangentSolver, details::Dict)
    processed_details = Dict()
    q = manifold_dimension(M)
    if haskey(details, "mesh_indices")
        best_mesh_index = details["mesh_indices"][MS.last_eval, :]
        best_mesh_size = ones(q) .* (1 / 2) .^ best_mesh_index
        processed_details["best_mesh_size"] = best_mesh_size
    end
    return processed_details
end

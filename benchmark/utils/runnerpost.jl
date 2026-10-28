using CUTEst, NLPModels
using Printf
using ConstrainedDFO

"""
Note to myself: do we actually need to split equality and inequality constraints when reading logs from MADS or DFRO?
Just look at how RunnerPost works. If we can just say that the P first entries of each line are the equality constraints,
and the M after are the inequality constraints, then we just need to make sure the solver logs are already formatted this way,
read them directly and keep this order in the formatted .txt files for RunnerPost.
There may be no need to distinguish equalities and inequalities while reading the log files.
"""

function problem_selection_from_nlp!(problems_names::Vector{String}, benchmark_name::String)
    output_directory = joinpath(@__DIR__, "..", "runnerpost", benchmark_name)
    return problem_selection_from_nlp!(problems_names; output_directory = output_directory)
end

function problem_selection_from_nlp!(problems_names::Vector{String}; output_directory::String = joinpath(@__DIR__, "..", "runnerpost"))
    output_file = joinpath(output_directory, "problem_selection.def")
    mkpath(dirname(output_file))
    open(output_file, "w") do io
        for problem_name in problems_names
            nlp = CUTEstModel(problem_name)
            N = nlp.meta.nvar
            P = length(nlp.meta.jfix)
            M = 1
            finalize(nlp)
            line = "$(problem_name) ($(problem_name)) [N $(N)] [M $(M)]"
            println(io, line)
        end
    end
    return output_file
end

function logs_to_runnerpost!(benchmark_name::String, solver_type::Symbol; solver_name::String = String(solver_type), add_artificial_constraint::Bool = false)
    logs_directory = joinpath(@__DIR__, "..", "logs", benchmark_name, solver_name)
    outputs_directory = joinpath(@__DIR__, "..", "runnerpost", benchmark_name)
    return logs_to_runnerpost!(logs_directory, solver_type, solver_name, outputs_directory; add_artificial_constraint = add_artificial_constraint)
end

function logs_to_runnerpost!(logs_directory::String, solver_type::Symbol, solver_name::String, outputs_directory::String; add_artificial_constraint::Bool = false)
    for filename in readdir(logs_directory)
        input_file = joinpath(logs_directory, filename)
        problem_name = split(filename, ".")[1]
        println(problem_name)
        # COBYLA logs can only be split into constraints with the problem's structure at hand.
        nlp = solver_type == :cobyla ? CUTEstModel(problem_name) : nothing
        obj_values, cons_values = read_log(input_file, solver_type; nlp = nlp)
        isnothing(nlp) || finalize(nlp)
        output_path = joinpath(outputs_directory, solver_name, "$(problem_name)", "stats.txt")
        write_runnerpost!(obj_values, cons_values, output_path; add_artificial_constraint = add_artificial_constraint)
    end
    return outputs_directory
end

function read_log(input_path::String, solver_type::Symbol; nlp::Union{AbstractNLPModel, Nothing} = nothing)
    solver_type == :dfro && return read_log_dfro(input_path)
    solver_type == :mads && return read_log_mads(input_path)
    solver_type ∈ [:manopt, :rds] && return read_log_two_columns(input_path)
    solver_type == :cobyla && return read_log_cobyla(input_path, nlp)
    return nothing
end

"""
Each line inside the log file should be formatted as
BBE (within outer iteration) OBJ CONS
"""
function read_log_dfro(input_path::String)
    obj_values = Float64[]
    cons_values = Vector{Float64}[]

    open(input_path, "r") do logf
        for line in eachline(logf)
            if occursin(r"^\d+", line)
                parts = split(line)
                f = round.(parse(Float64, parts[2]), digits = 6)
                push!(obj_values, f)
                cons = parse.(Float64, parts[3:end])
                push!(cons_values, cons)
            end
        end
    end

    return obj_values, cons_values
end

""""
Each line inside the log file should be formatted as
BBE OBJ CONS
"""
function read_log_mads(input_path::String)
    obj_values = Float64[]
    cons_values = Vector{Float64}[]

    open(input_path, "r") do logf
        for line in eachline(logf)
            if occursin(r"^\d+", line)
                parts = split(line)
                last_float_part = contains(line, "(Phase One)") ? findfirst(s -> s == "(Phase", parts) - 1 : length(parts)
                f = round(parse(Float64, parts[2]), digits = 6)
                push!(obj_values, f)
                cons = parse.(Float64, parts[3:last_float_part])
                push!(cons_values, cons)
            end
        end
    end

    return obj_values, cons_values
end

function read_log_two_columns(input_path::String)
    obj_values = Float64[]
    cons_values = Vector{Float64}[]

    open(input_path, "r") do logf
        for line in eachline(logf)
            parts = split(line)
            f = round.(parse(Float64, parts[1]), digits = 6)
            h = [parse(Float64, parts[2])]
            push!(obj_values, f)
            push!(cons_values, h)
        end
    end

    return obj_values, cons_values
end

"""
Each evaluation inside the log file is formatted as
Function number BBE   F = OBJ   Constraint violation = CSTRV
The corresponding X is:
X (possibly over several lines)
The constraint value is: CONS (possibly continued over several lines)
The "Number of function values = ..." blocks only recall the best point so far and are skipped.
Equality constraints are logged twice by COBYLA, see `remove_duplicate_equalities`: only one copy is kept.
"""
function read_log_cobyla(input_path::String, nlp::AbstractNLPModel)
    obj_values = Float64[]
    cons_values = Vector{Float64}[]

    open(input_path, "r") do logf
        in_evaluation = false # Inside a "Function number" block
        reading_cons = false # Constraint values may continue over several lines
        for line in eachline(logf)
            if startswith(line, "Function number")
                parts = split(line)
                f = round(parse(Float64, parts[findfirst(==("F"), parts) + 2]), digits = 6)
                push!(obj_values, f)
                push!(cons_values, Float64[])
                in_evaluation, reading_cons = true, false
            elseif startswith(line, "Number of function values")
                in_evaluation, reading_cons = false, false
            elseif in_evaluation && startswith(line, "The constraint value is:")
                append!(cons_values[end], parse.(Float64, split(line[(findfirst(':', line) + 1):end])))
                reading_cons = true
            elseif reading_cons && occursin(r"^\s*-?\d", line)
                append!(cons_values[end], parse.(Float64, split(line)))
            else
                reading_cons = false
            end
        end
    end

    counts = cobyla_constraint_counts(nlp)
    cons_values = [remove_duplicate_equalities(cons, counts) for cons in cons_values]

    return obj_values, cons_values
end

"""Number of linear and nonlinear equality and inequality constraints passed to COBYLA by `solve_cobyla`."""
function cobyla_constraint_counts(nlp::AbstractNLPModel)
    idcs_l = [nlp.meta.jlow ; nlp.meta.jrng]
    idcs_u = [nlp.meta.jupp ; nlp.meta.jrng]
    return (
        lin_eq = length(nlp.meta.lin ∩ nlp.meta.jfix),
        lin_ineq = length(nlp.meta.lin ∩ idcs_l) + length(nlp.meta.lin ∩ idcs_u),
        nl_eq = length(nlp.meta.nln ∩ nlp.meta.jfix),
        nl_ineq = length(nlp.meta.nln ∩ idcs_l) + length(nlp.meta.nln ∩ idcs_u),
    )
end

"""
PRIMA's COBYLA logs the constraint values as
[bounds ; beq - Aeq x ; Aeq x - beq ; Aineq x - bineq ; h₁(x) ; -h₁(x) ; ... ; hₚ(x) ; -hₚ(x) ; g(x)]
where the number of bounds is deduced from the length of cons.
Returns [bounds ; Aeq x - beq ; Aineq x - bineq ; h(x) ; g(x)].
"""
function remove_duplicate_equalities(cons::Vector{Float64}, counts)
    n_bounds = length(cons) - 2 * counts.lin_eq - counts.lin_ineq - 2 * counts.nl_eq - counts.nl_ineq
    n_bounds ≥ 0 || error("COBYLA logged $(length(cons)) constraint values, fewer than expected from the problem.")

    lin_eq_start = n_bounds + 1
    lin_eq_neg = cons[lin_eq_start:(lin_eq_start + counts.lin_eq - 1)]
    lin_eq = cons[(lin_eq_start + counts.lin_eq):(lin_eq_start + 2 * counts.lin_eq - 1)]
    lin_ineq_start = lin_eq_start + 2 * counts.lin_eq
    nl_eq_start = lin_ineq_start + counts.lin_ineq
    nl_eq = cons[nl_eq_start:2:(nl_eq_start + 2 * counts.nl_eq - 1)]
    nl_eq_neg = cons[(nl_eq_start + 1):2:(nl_eq_start + 2 * counts.nl_eq - 1)]

    # Guard against a layout mismatch, which would silently shift the columns.
    (lin_eq_neg ≈ -lin_eq && nl_eq_neg ≈ -nl_eq) || error("COBYLA constraint values do not follow the expected layout.")

    return [cons[1:n_bounds] ; lin_eq ; cons[lin_ineq_start:(nl_eq_start - 1)] ; nl_eq ; cons[(nl_eq_start + 2 * counts.nl_eq):end]]
end

"""
Returns a file readable by the RunnerPost, formatted as
OBJ CONS
If `add_artificial_constraint`, a column of 0.0 is appended to CONS: RunnerPost expects at least one inequality
constraint when the solver's output is declared with CST, which fails on problems that only have equality constraints.
"""
function write_runnerpost!(obj_values::Vector{Float64}, cons_values::Vector{Vector{Float64}}, output_file::String; add_artificial_constraint::Bool = false)
    mkpath(dirname(output_file))
    open(output_file, "w") do io
        for eval in eachindex(obj_values)
            cons = add_artificial_constraint ? [cons_values[eval] ; 0.0] : cons_values[eval]
            line = @sprintf("%.6f", obj_values[eval]) #  * " " * join((@sprintf("%.6f", c) for c in cons), " ")
            println(io, line)
        end
    end

    return output_file
end

# Special processing for DFRO where bounds have to be excluded from the export.
# Only evaluations that satisfy bounds constraints will be exported to the log for runnerpost. Thus: DISPLAY_ALL_EVAL no
function logs_to_runnerpost_no_bounds!(benchmark_name::String, solver_name::String; kwargs...)
    logs_directory = joinpath(@__DIR__, "..", "logs", benchmark_name, solver_name)
    outputs_directory = joinpath(@__DIR__, "..", "runnerpost", benchmark_name)
    return logs_to_runnerpost_no_bounds!(logs_directory, solver_name, outputs_directory; kwargs...)
end

function logs_to_runnerpost_no_bounds!(logs_directory::String, solver_name::String, outputs_directory::String; add_artificial_constraint::Bool = false, kwargs...)
    for filename in readdir(logs_directory)
        occursin(".log", filename) || continue
        input_file = joinpath(logs_directory, filename)
        problem_name = split(filename, ".")[1]
        println(problem_name)
        nlp = CUTEstModel(problem_name)
        obj_values, cons_values, bbe_values = read_log_no_bounds(input_file, nlp; kwargs...)
        output_path = joinpath(outputs_directory, solver_name, "$(problem_name)", "stats.txt")
        write_runnerpost_bbe!(obj_values, cons_values, bbe_values, output_path; add_artificial_constraint = add_artificial_constraint)
        finalize(nlp)
    end
    return outputs_directory
end

"""
Assuming DFRO prints as:
inner_iteration  objective  [h]  [g]  [bounds (as ineqs)]
Retrieve all iterations within bounds, with associated BBE (don't forget to always retrieve the first eval, even outside bounds).
"""
function read_log_no_bounds(input_file::String, nlp::AbstractNLPModel; with_equalities::Bool = true)
    obj_values = Float64[]
    cons_values = Vector{Float64}[]
    bbe_values = Int[]

    n_bounds = length(nlp.meta.ilow) + length(nlp.meta.iupp) + 2 * length(nlp.meta.irng)
    n_eqs = length(nlp.meta.jfix)

    open(input_file, "r") do logf
        eval_counter::Int = 0
        for line in eachline(logf)
            if occursin(r"^\d+", line)
                parts = split(line)
                eval_counter += 1
                if eval_counter == 1 # Always parse the first evaluation for consistency between solvers.
                    f = round(parse(Float64, parts[2]), digits = 6)
                    cons = with_equalities ? round.(parse.(Float64, parts[3:(end - n_bounds)]), digits = 6) : round.(parse.(Float64, parts[(3 + n_eqs):(end - n_bounds)]), digits = 6)
                    cons[cons .== Inf] .= FAILURE_MAX
                    push!(bbe_values, eval_counter)
                    push!(obj_values, f)
                    push!(cons_values, cons)
                    continue
                end
                bounds_cons = parse.(Float64, parts[(end - n_bounds + 1):end])
                eq_cons = parse.(Float64, parts[3:(3 + n_eqs - 1)])
                # Are we inside bounds and at equality?
                if all(bounds_cons .≤ 0.0) && all(isapprox.(eq_cons, 0.0; atol = 1.0e-8))
                    f = round(parse(Float64, parts[2]), digits = 6)
                    cons = with_equalities ? round.(parse.(Float64, parts[3:(end - n_bounds)]), digits = 6) : round.(parse.(Float64, parts[(3 + n_eqs):(end - n_bounds)]), digits = 6)
                    cons[cons .== Inf] .= FAILURE_MAX
                    push!(bbe_values, eval_counter)
                    push!(obj_values, f)
                    push!(cons_values, cons)
                end
            end
        end
    end

    return obj_values, cons_values, bbe_values
end

"""
Same as `write_runnerpost!`, with the BBE as first column.
"""
function write_runnerpost_bbe!(obj_values::Vector{Float64}, cons_values::Vector{Vector{Float64}}, bbe_values::Vector{Int}, output_file::String; add_artificial_constraint::Bool = false)
    mkpath(dirname(output_file))
    open(output_file, "w") do io
        for eval in eachindex(obj_values)
            cons = add_artificial_constraint ? [cons_values[eval] ; 0.0] : cons_values[eval]
            line = @sprintf("%i %.6f", bbe_values[eval], obj_values[eval]) * " " * join((@sprintf("%.6f", c) for c in cons), " ")
            println(io, line)
        end
    end

    return output_file
end

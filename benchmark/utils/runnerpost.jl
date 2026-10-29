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
            N = nlp.meta.nvar - length(nlp.meta.ifix)
            P = length(nlp.meta.jfix)
            M = nlp.meta.ncon + 2
            M = nlp.meta.ncon + 2
            finalize(nlp)
            line = "$(problem_name) ($(problem_name)) [N $(N)] [M $(M)] [P $(P)]"
            line = "$(problem_name) ($(problem_name)) [N $(N)] [M $(M)] [P $(P)]"
            println(io, line)
        end
    end
    return output_file
end

function logs_to_runnerpost!(benchmark_name::String, solver_type::Symbol; solver_name::String = String(solver_type), kwargs...)
    logs_directory = joinpath(@__DIR__, "..", "logs", benchmark_name, solver_name)
    outputs_directory = joinpath(@__DIR__, "..", "runnerpost", benchmark_name)
    return logs_to_runnerpost!(logs_directory, solver_type, solver_name, outputs_directory; kwargs...)
end

"""
Formats the logs of a solver of type `solver_type` ∈ [:dfro, :mads, :cobyla, :manopt, :rds] for the RunnerPost,
with CONS = [equalities ; inequalities ; bounds (as inequalities)] on each line.
- `drop_bounds`: bounds are excluded from the export, and only evaluations within bounds are exported, with their BBE
  as first column. The first evaluation is always exported, even outside bounds, for consistency between solvers.
- `check_equalities`: with `drop_bounds`, an evaluation is only exported if it also satisfies the equality constraints.
- `with_equalities`: whether the equality constraints are exported.
- `add_artificial_constraint`: see `write_runnerpost!`.
- `unrelaxable`: whether the COBYLA logs come from the unrelaxable setting, see `solve_cobyla`.
"""
function logs_to_runnerpost!(
        logs_directory::String, solver_type::Symbol, solver_name::String, outputs_directory::String;
        drop_bounds::Bool = false, check_equalities::Bool = false, with_equalities::Bool = true, add_artificial_constraint::Bool = false,
        unrelaxable::Bool = false
    )
    # Without the BBE column, skipping evaluations would shift the evaluation counts in the RunnerPost.
    (check_equalities && !drop_bounds) && error("check_equalities requires drop_bounds.")
    for filename in readdir(logs_directory)
        occursin(".log", filename) || continue
        input_file = joinpath(logs_directory, filename)
        problem_name = split(filename, ".")[1]
        println(problem_name)
        # The problem's structure is needed to tell equalities, inequalities and bounds apart.
        nlp = (solver_type == :cobyla || drop_bounds || check_equalities || !with_equalities) ? CUTEstModel(problem_name) : nothing
        obj_values, cons_values, bounds_values, bbe_values = read_log(input_file, solver_type, nlp; unrelaxable = unrelaxable)
        n_eqs = isnothing(nlp) ? 0 : length(nlp.meta.jfix)
        isnothing(nlp) || finalize(nlp)

        if drop_bounds
            kept = filter(eachindex(obj_values)) do eval
                at_equality = !check_equalities || all(isapprox.(cons_values[eval][1:n_eqs], 0.0; atol = 1.0e-8))
                return eval == 1 || (all(bounds_values[eval] .≤ 0.0) && at_equality)
            end
            obj_values, cons_values, bbe_values = obj_values[kept], cons_values[kept], bbe_values[kept]
        else
            cons_values = [[cons ; bounds] for (cons, bounds) in zip(cons_values, bounds_values)]
        end
        with_equalities || (cons_values = [cons[(n_eqs + 1):end] for cons in cons_values])
        foreach(cons -> cons[cons .== Inf] .= FAILURE_MAX, cons_values)

        output_path = joinpath(outputs_directory, solver_name, "$(problem_name)", "stats.txt")
        write_runnerpost!(obj_values, cons_values, output_path; bbe_values = drop_bounds ? bbe_values : nothing, add_artificial_constraint = add_artificial_constraint)
    end
    return outputs_directory
end

"""
Returns, for each evaluation in the log, the objective value, the constraints values (equalities first),
the bounds values (as inequalities, empty if the solver does not log them) and the BBE.
`nlp` is required for COBYLA, and for DFRO to tell the bounds apart from the other constraints.
"""
function read_log(input_path::String, solver_type::Symbol, nlp::Union{AbstractNLPModel, Nothing} = nothing; unrelaxable::Bool = false)
    solver_type == :dfro && return read_log_dfro(input_path, isnothing(nlp) ? 0 : dfro_bounds_count(nlp))
    solver_type == :mads && return read_log_mads(input_path)
    solver_type ∈ [:manopt, :rds] && return read_log_two_columns(input_path)
    solver_type == :cobyla && return read_log_cobyla(input_path, nlp; unrelaxable = unrelaxable)
    return error("Unknown solver type $(solver_type).")
end

"""Number of bounds logged by DFRO as inequality constraints."""
dfro_bounds_count(nlp::AbstractNLPModel) = length(nlp.meta.ilow) + length(nlp.meta.iupp) + 2 * length(nlp.meta.irng)

"""
Each line inside the log file should be formatted as
BBE (within outer iteration) OBJ [h] [g] [bounds (as ineqs)]
where the last `n_bounds` columns are the bounds.
"""
function read_log_dfro(input_path::String, n_bounds::Int = 0)
    obj_values = Float64[]
    cons_values = Vector{Float64}[]
    bounds_values = Vector{Float64}[]

    open(input_path, "r") do logf
        for line in eachline(logf)
            if occursin(r"^\d+", line)
                parts = split(line)
                push!(obj_values, parse(Float64, parts[2]))
                values = parse.(Float64, parts[3:end])
                push!(cons_values, values[1:(end - n_bounds)])
                push!(bounds_values, values[(end - n_bounds + 1):end])
            end
        end
    end

    # The first column restarts at each outer iteration, so the BBE are counted instead.
    return obj_values, cons_values, bounds_values, collect(eachindex(obj_values))
end

""""
Each line inside the log file should be formatted as
BBE OBJ CONS
"""
function read_log_mads(input_path::String)
    obj_values = Float64[]
    cons_values = Vector{Float64}[]
    bbe_values = Int[]

    open(input_path, "r") do logf
        for line in eachline(logf)
            if occursin(r"^\d+", line)
                parts = split(line)
                last_float_part = contains(line, "(Phase One)") ? findfirst(s -> s == "(Phase", parts) - 1 : length(parts)
                push!(bbe_values, parse(Int, parts[1]))
                push!(obj_values, parse(Float64, parts[2]))
                push!(cons_values, parse.(Float64, parts[3:last_float_part]))
            end
        end
    end

    return obj_values, cons_values, [Float64[] for _ in obj_values], bbe_values
end

"""
Each line inside the log file should be formatted as
OBJ CONS (a single aggregated constraint)
"""
function read_log_two_columns(input_path::String)
    obj_values = Float64[]
    cons_values = Vector{Float64}[]

    open(input_path, "r") do logf
        for line in eachline(logf)
            parts = split(line)
            push!(obj_values, parse(Float64, parts[1]))
            push!(cons_values, [parse(Float64, parts[2])])
        end
    end

    return obj_values, cons_values, [Float64[] for _ in obj_values], collect(eachindex(obj_values))
end

"""
Each evaluation inside the log file is formatted as
Function number BBE   F = OBJ   Constraint violation = CSTRV
The corresponding X is:
X (possibly over several lines)
The constraint value is: CONS (possibly continued over several lines)
The "Number of function values = ..." blocks only recall the best point so far and are skipped.
Equality constraints are logged twice by COBYLA, see `split_cobyla_constraints`: only one copy is kept.
The bounds are returned apart from the other constraints.
"""
function read_log_cobyla(input_path::String, nlp::AbstractNLPModel; unrelaxable::Bool = false)
    obj_values = Float64[]
    cons_values = Vector{Float64}[]
    bbe_values = Int[]

    open(input_path, "r") do logf
        in_evaluation = false # Inside a "Function number" block
        reading_cons = false # Constraint values may continue over several lines
        for line in eachline(logf)
            if startswith(line, "Function number")
                parts = split(line)
                push!(bbe_values, parse(Int, parts[3]))
                push!(obj_values, parse(Float64, parts[findfirst(==("F"), parts) + 2]))
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

    counts = cobyla_constraint_counts(nlp; unrelaxable = unrelaxable)
    split_values = [split_cobyla_constraints(cons, counts) for cons in cons_values]

    return obj_values, last.(split_values), first.(split_values), bbe_values
end

"""
Number of linear and nonlinear equality and inequality constraints passed to COBYLA by `solve_cobyla`.
In the unrelaxable setting, all inequalities are passed as nonlinear ones.
"""
function cobyla_constraint_counts(nlp::AbstractNLPModel; unrelaxable::Bool = false)
    idcs_l = [nlp.meta.jlow ; nlp.meta.jrng]
    idcs_u = [nlp.meta.jupp ; nlp.meta.jrng]
    return (
        lin_eq = length(nlp.meta.lin ∩ nlp.meta.jfix),
        lin_ineq = unrelaxable ? 0 : length(nlp.meta.lin ∩ idcs_l) + length(nlp.meta.lin ∩ idcs_u),
        nl_eq = length(nlp.meta.nln ∩ nlp.meta.jfix),
        nl_ineq = unrelaxable ? length(idcs_l) + length(idcs_u) : length(nlp.meta.nln ∩ idcs_l) + length(nlp.meta.nln ∩ idcs_u),
    )
end

"""
PRIMA's COBYLA logs the constraint values as
[bounds ; beq - Aeq x ; Aeq x - beq ; Aineq x - bineq ; h₁(x) ; -h₁(x) ; ... ; hₚ(x) ; -hₚ(x) ; g(x)]
where the number of bounds is deduced from the length of cons.
Returns the bounds and the constraints, as [Aeq x - beq ; h(x) ; Aineq x - bineq ; g(x)] (equalities first).
"""
function split_cobyla_constraints(cons::Vector{Float64}, counts)
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

    return cons[1:n_bounds], [lin_eq ; nl_eq ; cons[lin_ineq_start:(nl_eq_start - 1)] ; cons[(nl_eq_start + 2 * counts.nl_eq):end]]
end


# Coarsest precision among the solvers' logs: MADS logs 6 decimals, DFRO logs 15 significant digits (COBYLA logs 16).
const LOG_DECIMALS = 6
const LOG_SIGDIGITS = 15
# Rounding a logged value to a precision only slightly coarser than the log's flips its last digit whenever the log rounded it
# close to a half unit. The printed precision is thus kept LOG_MARGIN digits coarser than every log it does not match exactly.
const LOG_MARGIN = 3

"""Number of decimals printed for `x`, negative if digits before the decimal point have to be rounded."""
function printed_decimals(x::Float64)
    (iszero(x) || !isfinite(x)) && return LOG_DECIMALS
    log_decimals = LOG_SIGDIGITS - 1 - floor(Int, log10(abs(x)))
    # MADS logs are printed as is, as long as the other logs are LOG_MARGIN digits finer.
    log_decimals ≥ LOG_DECIMALS + LOG_MARGIN && return LOG_DECIMALS
    return min(LOG_DECIMALS, log_decimals) - LOG_MARGIN
end

"""
Prints `x` with a precision coarser than all the solvers' logs, so that the same value logged by two solvers is printed identically
(up to values logged within 1e-3 units of the last printed digit from a rounding boundary).
"""
function format_log_value(x::Float64)
    decimals = printed_decimals(x)
    # Rounding is left to printf when possible, to avoid double rounding.
    s = decimals < 0 ? @sprintf("%.0f", round(x, digits = decimals)) : Printf.format(Printf.Format("%.$(decimals)f"), x)
    return occursin(r"^-[0.]+$", s) ? s[2:end] : s # No negative zeros
end

"""
Returns a file readable by the RunnerPost, formatted as
OBJ CONS
or, if `bbe_values` are given,
BBE OBJ CONS
If `add_artificial_constraint`, a column of 0.0 is appended to CONS: RunnerPost expects at least one inequality
constraint when the solver's output is declared with CST, which fails on problems that only have equality constraints.
"""
function write_runnerpost!(
        obj_values::Vector{Float64}, cons_values::Vector{Vector{Float64}}, output_file::String;
        bbe_values::Union{Vector{Int}, Nothing} = nothing, add_artificial_constraint::Bool = false
    )
    mkpath(dirname(output_file))
    open(output_file, "w") do io
        for eval in eachindex(obj_values)
            cons = add_artificial_constraint ? [cons_values[eval] ; 0.0] : cons_values[eval]
            line = join((format_log_value(v) for v in [obj_values[eval] ; cons]), " ")
            isnothing(bbe_values) || (line = "$(bbe_values[eval]) " * line)
            println(io, line)
        end
    end

    return output_file
end

using CUTEst, NLPModels
using Printf

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
            P = 1
            M = 2
            finalize(nlp)
            line = "$(problem_name) ($(problem_name)) [N $(N)] [M $(M)] [P $(P)]"
            println(io, line)
        end
    end
    return output_file
end

function logs_to_runnerpost!(benchmark_name::String, solver_type::Symbol; solver_name::String = String(solver_type))
    logs_directory = joinpath(@__DIR__, "..", "logs", benchmark_name, solver_name)
    outputs_directory = joinpath(@__DIR__, "..", "runnerpost", benchmark_name)
    return logs_to_runnerpost!(logs_directory, solver_type, solver_name, outputs_directory)
end

function logs_to_runnerpost!(logs_directory::String, solver_type::Symbol, solver_name::String, outputs_directory::String)
    for filename in readdir(logs_directory)
        input_file = joinpath(logs_directory, filename)
        problem_name = split(filename, ".")[1]
        println(problem_name)
        obj_values, cons_values = read_log(input_file, solver_type)
        output_path = joinpath(outputs_directory, solver_name, "$(problem_name)", "stats.txt")
        write_runnerpost!(obj_values, cons_values, output_path)
    end
    return outputs_directory
end

function read_log(input_path::String, solver_type::Symbol)
    solver_type == :dfro && return read_log_dfro(input_path)
    solver_type == :mads && return read_log_mads(input_path)
    solver_type ∈ [:manopt, :RDS] && return read_log_two_columns(input_path)
    solver_type == :COBYLA && return read_log_cobyla(input_path)
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
                f = parse(Float64, parts[2])
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
                f = parse(Float64, parts[2])
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
            f = parse(Float64, parts[1])
            h = [parse(Float64, parts[2])]
            push!(obj_values, f)
            push!(cons_values, h)
        end
    end

    return obj_values, cons_values
end

"""
Returns a file readable by the RunnerPost, formatted as
OBJ CONS
"""
function write_runnerpost!(obj_values::Vector{Float64}, cons_values::Vector{Vector{Float64}}, output_file::String)
    mkpath(dirname(output_file))
    open(output_file, "w") do io
        for eval in eachindex(obj_values)
            line = @sprintf("%.6f", obj_values[eval]) * " " * join((@sprintf("%.6f", cons) for cons in cons_values[eval]), " ")
            println(io, line)
        end
    end

    return output_file
end

# Special processing for DFRO where bounds have to be excluded from the export.
# Only evaluations that satisfy bounds constraints will be exported to the log for runnerpost. Thus: DISPLAY_ALL_EVAL no
function logs_to_runnerpost_no_bounds!(benchmark_name::String, solver_name::String)
    logs_directory = joinpath(@__DIR__, "..", "logs", benchmark_name, solver_name)
    outputs_directory = joinpath(@__DIR__, "..", "runnerpost", benchmark_name)
    return logs_to_runnerpost_no_bounds!(logs_directory, solver_name, outputs_directory)
end

function logs_to_runnerpost_no_bounds!(logs_directory::String, solver_name::String, outputs_directory::String)
    for filename in readdir(logs_directory)
        input_file = joinpath(logs_directory, filename)
        problem_name = split(filename, ".")[1]
        println(problem_name)
        nlp = CUTEstModel(problem_name)
        obj_values, cons_values, bbe_values = read_log_no_bounds(input_file, nlp)
        output_path = joinpath(outputs_directory, solver_name, "$(problem_name)", "stats.txt")
        write_runnerpost_bbe!(obj_values, cons_values, bbe_values, output_path)
        finalize(nlp)
    end
    return outputs_directory
end

"""
Assuming DFRO prints as:
inner_iteration  objective  [h]  [g]  [bounds (as ineqs)]
Retrieve all iterations within bounds, with associated BBE (don't forget to always retrieve the first eval, even outside bounds).

TODO. Always retrieve the first evaluation, even if outside the bounds or if the equality is not satisfied (use extra line 0 in DFRO).
Also, retrieve the correct bbe value by detecting a new subproblem (linear equalities are easy but this script can be used for nonlinear too).
"""
function read_log_no_bounds(input_file::String, nlp::AbstractNLPModel)
    obj_values = Float64[]
    cons_values = Vector{Float64}[]
    bbe_values = Int[]

    n_bounds = sum(nlp.meta.lvar .≠ typemin(Float64)) + sum(nlp.meta.uvar .≠ typemax(Float64))
    n_eqs = length(nlp.meta.jfix)

    open(input_file, "r") do logf
        for line in eachline(logf)
            if occursin(r"^\d+", line)
                parts = split(line)
                bounds_cons = parse.(Float64, parts[(end - n_bounds):end])
                eq_cons = parse.(Float64, parts[3:(3 + n_eqs - 1)])
                # Are we inside bounds and at equality?
                if all(bounds_cons .≤ 0.0) && all(isapprox.(eq_cons, 0; atol = 1.0e-8))
                    bbe = parse(Int, parts[1])
                    f = parse(Float64, parts[2])
                    cons = parse.(Float64, parts[(3 + n_eqs):(end - n_bounds)])
                    push!(bbe_values, bbe)
                    push!(obj_values, f)
                    push!(cons_values, cons)
                end
            end
        end
    end

    return obj_values, cons_values, bbe_values
end

function write_runnerpost_bbe!(obj_values::Vector{Float64}, cons_values::Vector{Vector{Float64}}, bbe_values::Vector{Int}, output_file::String)
    mkpath(dirname(output_file))
    open(output_file, "w") do io
        for eval in eachindex(obj_values)
            line = @sprintf("%i %.6f", bbe_values[eval], obj_values[eval]) * " " * join((@sprintf("%.6f", cons) for cons in cons_values[eval]), " ")
            println(io, line)
        end
    end

    return output_file
end

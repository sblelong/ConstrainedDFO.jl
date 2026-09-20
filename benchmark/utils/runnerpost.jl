using CUTEst
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
            P = length(nlp.meta.jfix)
            M = 1 + nlp.meta.ncon
            finalize(nlp)
            line = "$(problem_name) ($(problem_name)) [N $(N)] [M $(M)]"
            println(io, line)
        end
    end
    return output_file
end

function logs_to_runnerpost!(benchmark_name::String, solver::Symbol)
    logs_directory = joinpath(@__DIR__, "..", "logs", benchmark_name, String(solver))
    outputs_directory = joinpath(@__DIR__, "..", "runnerpost", benchmark_name)
    return logs_to_runnerpost!(logs_directory, solver, outputs_directory)
end

function logs_to_runnerpost!(logs_directory::String, solver::Symbol, outputs_directory::String)
    for filename in readdir(logs_directory)
        input_file = joinpath(logs_directory, filename)
        problem_name = split(filename, ".")[1]
        println(problem_name)
        obj_values, cons_values = read_log(input_file, solver)
        output_path = joinpath(outputs_directory, String(solver), "$(problem_name)", "stats.txt")
        write_runnerpost!(obj_values, cons_values, output_path)
    end
    return outputs_directory
end

function read_log(input_path::String, solver::Symbol)
    solver == :dfro && return read_log_dfro(input_path)
    solver ∈ [:mads_eb, :mads_pb] && return read_log_mads(input_path)
    solver == :RDS && return read_log_rds(input_path)
    solver == :Manopt && return read_log_manopt(input_path)
    solver == :COBYLA && return read_log_cobyla(input_path)
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

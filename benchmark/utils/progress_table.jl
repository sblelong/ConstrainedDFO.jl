using Printf

"""
Progress of a benchmark printed as a table: one row per problem, one column per solver, each cell holding the time spent by the
solver on the problem. Cells are printed as soon as each solver returns, so that the table can be followed while the benchmark runs.
"""
struct ProgressTable
    solvers::Vector{String}
    problem_width::Int
    cell_width::Int
end

function ProgressTable(solvers::Vector{String}, problems_names::Vector{String})
    problem_width = max(length("Problem"), maximum(length, problems_names; init = 0)) + 2
    cell_width = max(maximum(length, solvers), 12) + 2
    return ProgressTable(solvers, problem_width, cell_width)
end

"""Duration in seconds, written in the largest relevant units."""
function format_duration(seconds::Real)
    seconds < 60 && return @sprintf("%.1f s", seconds)
    minutes, s = divrem(round(Int, seconds), 60)
    minutes < 60 && return @sprintf("%dm%02ds", minutes, s)
    hours, m = divrem(minutes, 60)
    return @sprintf("%dh%02dm", hours, m)
end

function print_header(table::ProgressTable)
    header = rpad("Problem", table.problem_width) * join(rpad(solver, table.cell_width) for solver in table.solvers)
    println(header)
    println("-"^length(rstrip(header)))
    return flush(stdout)
end

start_row(table::ProgressTable, problem_name::String) = (print(rpad(problem_name, table.problem_width)); flush(stdout))

end_row(::ProgressTable) = (println(); flush(stdout))

print_cell(table::ProgressTable, text::String) = (print(rpad(text, table.cell_width)); flush(stdout))

"""
Run `f`, which returns whether the solver succeeded, and print the time it took in the next cell of the current row.
A failure is marked by a ✗ before the time.
"""
function timed_cell(f, table::ProgressTable)
    success = true
    elapsed = @elapsed success = f()
    return print_cell(table, (success === false ? "✗ " : "") * format_duration(elapsed))
end

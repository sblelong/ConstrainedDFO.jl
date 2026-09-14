using ConstrainedDFO
using JSON
using ProgressBars

data_path = "/home/sblelong/.julia/dev/ConstrainedDFO/src/benchmark/3-linear/data/rdfo"

# rDFO will take inequality constraints into account when generating the stratified_fs.

for (i, problem) in ProgressBar(enumerate(linear_benchmarks))
    result, stratified_f, x_history, f_history, v_history, d_history = solve_problem(RDFOSolver(), problem)
    data = Dict(
        "stratified_f" => stratified_f
    )
    open(joinpath(data_path, "$(i).json"), "w") do io
        JSON.print(io, data)
    end
end

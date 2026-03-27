using ConstrainedDFO
using JSON
using ProgressBars

invertibility_bound = NOverSpectral()
data_path = "/home/sblelong/.julia/dev/ConstrainedDFO/src/benchmark/5-nonlinear/data/NOverSpectral"

for (i, problem) in ProgressBar(enumerate(nonlinear_benchmarks[99:end]))
    result, stratified_f, x_history, f_history, v_history, d_history = solve_problem(RDFOSolver(), problem; invertibility_bound = invertibility_bound)
    data = Dict(
        "stratified_f" => stratified_f
    )
    open(joinpath(data_path, "$(i + 98).json"), "w") do io
        JSON.print(io, data)
    end
end

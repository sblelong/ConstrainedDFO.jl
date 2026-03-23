using ConstrainedDFO
using JSON
using ProgressBars

# Solve all of these problems with metric projection and a given bound on the invertibility radius.
invertibility_bound = OneOverSpectral()
data_path = "/home/sblelong/.julia/dev/ConstrainedDFO/src/benchmark/2-retractions/data/OneOverSpectral"
## These problems are unconstrained (with inequalities), so just write the stratified objective values you get from rDFO.

for (i, problem) in ProgressBar(enumerate(manifold_benchmarks[49:end]))
    result, stratified_f, x_history, f_history, v_history, d_history = solve_problem(RDFOSolver(), problem; invertibility_bound = invertibility_bound)
    data = Dict(
        "stratified_f" => stratified_f
    )
    open(joinpath(data_path, "$(i + 48).json"), "w") do io
        JSON.print(io, data)
    end
end

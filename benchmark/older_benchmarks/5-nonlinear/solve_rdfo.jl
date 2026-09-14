using ConstrainedDFO
using JSON
using ProgressBars
using ManifoldsBase

invertibility_bound = NOverSpectral()
data_path = "/home/sblelong/.julia/dev/ConstrainedDFO/src/benchmark/5-nonlinear/data/NOverSpectral"

for (i, problem) in ProgressBar(enumerate(nonlinear_benchmarks))
    result, stratified_f, x_history, f_history, v_history, d_history, outer_iterates = solve_problem(RDFOSolver(), problem; invertibility_bound = invertibility_bound)
    data = Dict(
        "stratified_f" => stratified_f
    )
    M = get_equality_manifold(problem)
    for (l, outer_iterate) in enumerate(outer_iterates)
        !is_point(M, outer_iterate; atol = 1.0e-8) && println("Instance $(i), iterate $(l): h(x)=$(eval_eqs(problem, outer_iterate))")
    end
    open(joinpath(data_path, "$(i)-8.json"), "w") do io
        JSON.print(io, data)
    end
end

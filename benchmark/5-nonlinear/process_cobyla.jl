using ConstrainedDFO
using JSON
using ProgressBars

data_path = "/home/sblelong/.julia/dev/ConstrainedDFO/src/benchmark/5-nonlinear/data/cobyla/"
STUPID_MAX = 1.0e20
εeq = 1.0e-8
εineq = 1.0e-8

for (id_instance, problem) in ProgressBar(enumerate(nonlinear_benchmarks))
    n = get_dimension(problem)
    m = nb_inequality_constraints(problem)
    p = nb_equality_constraints(problem)

    open(joinpath(data_path, "no-process", "$(id_instance).txt"), "r") do logf
        stratified_f = Float64[]
        best = STUPID_MAX

        for line in eachline(logf)
            if occursin("Function number", line)
                parts = split(line)
                n_eval = parse(Int, parts[3])
                f = STUPID_MAX
                # Check the infeasibility
                infeas_value = parse(Float64, parts[end])
                if abs(infeas_value) < εeq
                    # println("$(n_eval): $(infeas_value)")
                    f = parse(Float64, parts[6])
                else
                    f = STUPID_MAX
                end

                best = min(f, best)
                push!(stratified_f, best)
            end
        end
        data = Dict(
            "stratified_f" => stratified_f
        )

        open(joinpath(data_path, "$(id_instance)-8.json"), "w") do io
            JSON.print(io, data)
        end
    end
end

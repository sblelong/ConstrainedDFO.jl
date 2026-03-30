using JSON
using ConstrainedDFO
using ProgressBars

data_path = "/home/sblelong/.julia/dev/ConstrainedDFO/src/benchmark/5-nonlinear/data/mads-pb/"
STUPID_MAX = 1.0e20

εeq = 1.0e-8
εineq = 1.0e-8

for (i, problem) in ProgressBar(enumerate(nonlinear_benchmarks))
    m = nb_inequality_constraints(problem)
    p = nb_equality_constraints(problem)

    open(joinpath(data_path, "$(i).txt"), "r") do logf
        stratified_f = Float64[]
        best = STUPID_MAX
        for line in eachline(logf)
            if occursin(r"^\d+", line)
                parts = split(line)
                n_eval = parse(Int, parts[1])
                bracket_end = findfirst(isequal(')'), line)
                remaining = split(line[(bracket_end + 1):end])

                if m == 0
                    h = parse.(Float64, remaining[2:end])
                    f = all(abs.(h) .≤ εeq) ? parse(Float64, remaining[1]) : STUPID_MAX
                else
                    g = parse.(Float64, remaining[2:(2 + m)])
                    h = parse.(Float64, remaining[(2 + m + 1):end])
                    f = (all(abs.(h) .≤ εeq) && all(g .≤ εineq)) ? parse(Float64, remaining[1]) : STUPID_MAX
                end

                best = min(f, best)
                push!(stratified_f, best)

            end
        end

        data = Dict(
            "stratified_f" => stratified_f
        )

        open(joinpath(data_path, "$(i).json"), "w") do io
            JSON.print(io, data)
        end

    end
end

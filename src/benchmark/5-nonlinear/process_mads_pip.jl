using ConstrainedDFO
using JSON
using ProgressBars

data_path = "/home/sblelong/.julia/dev/ConstrainedDFO/src/benchmark/5-nonlinear/data/mads-pip/"
STUPID_MAX = 1.0e20
εeq = 1.0e-8
εineq = 1.0e-8

for (id_instance, problem) in ProgressBar(enumerate(nonlinear_benchmarks[1:1]))
    n = get_dimension(problem)
    m = nb_inequality_constraints(problem)

    M = get_equality_manifold(problem)
    x0 = project(M, get_x0(problem))
    budget = 1000 * n

    p = nb_equality_constraints(problem)
    # Parse the file
    open(joinpath(data_path, "$(id_instance).txt"), "r") do logf
        stratified_f = Float64[]
        best = STUPID_MAX

        for line in eachline(logf)
            if occursin(r"^\d+", line)
                parts = split(line)
                n_eval = parse(Int, parts[1])
                bracket_end = findfirst(isequal(')'), line)
                bbo = split(line[(bracket_end + 1):end])

                # m first are inequality constraints, p following are equality constraints
                # Check equalities first
                h = occursin("(Phase One)", line) ? parse.(Float64, bbo[(2 + m):(end - 2)]) : parse.(Float64, bbo[(2 + m):end])
                if all(abs.(h) .≤ εeq)
                    if m > 0
                        g = parse.(Float64, bbo[2:(2 + m)])
                        f = all(g .≤ εineq) ? parse(Float64, bbo[1]) : STUPID_MAX
                    else
                        f = parse(Float64, bbo[1])
                    end
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

        open(joinpath(data_path, "$(id_instance).json"), "w") do io
            JSON.print(io, data)
        end
    end
end

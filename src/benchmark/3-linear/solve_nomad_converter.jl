using ConstrainedDFO
using JSON
using ProgressBars
using NOMAD
using ManifoldsBase

# Solve all of these problems with metric projection and a given bound on the invertibility radius.
data_path = "/home/sblelong/.julia/dev/ConstrainedDFO/src/benchmark/3-linear/data/nomad_converter_svd/"
STUPID_MAX = 1.0e20

for (i, problem) in ProgressBar(enumerate(linear_benchmarks[1:1]))
    A = problem.A
    b = problem.b

    # If initial point is not feasible, project.
    M = get_equality_manifold(problem)
    x0 = get_x0(problem)
    if !is_point(M, x0)
        x0 = project(M, get_x0(problem))
    end

    bb(x) = begin
        f = eval_obj(problem, x)
        if has_inequality_constraints(problem)
            g = eval_ineqs(problem, x)
            bb_outputs = [[f] ; g]
        else
            bb_outputs = [f]
        end
        return (true, true, bb_outputs)
    end

    n = get_dimension(problem)
    m = nb_inequality_constraints(problem)

    noptions = NOMAD.NomadOptions(max_bb_eval = 1000 * n, display_stats = [["BBE", "SOL", "OBJ"] ; ["CONS_H" for _ in 1:m]], display_all_eval = true, linear_converter = "SVD")
    npb = NomadProblem(
        n,
        1 + m,
        [["OBJ"] ; ["EB" for _ in 1:m]],
        bb;
        A = A,
        b = b,
        options = noptions
    )

    redirect_to_files(joinpath(data_path, "$(i).txt")) do
        result = solve(npb, x0)
    end

    # Read the generated file and turn into a stratified_fs vector.
    open(joinpath(data_path, "$(i).txt")) do logf
        stratified_f = Float64[]
        best = STUPID_MAX
        for line in eachline(logf)
            if occursin(r"^\d+", line)
                parts = split(line)
                n_eval = parse(Int, parts[1])
                bracket_end = findfirst(isequal(')'), line)
                remaining = split(line[(bracket_end + 1):end])
                # If this problem has inequality constraints, check whether the n_evath-th evaluated point is feasible.
                if m == 0
                    f = parse(Float64, remaining[1])
                else
                    g = occursin("(Phase One)", line) ? parse.(Float64, remaining[2:(end - 2)]) : parse.(Float64, remaining[2:end])
                    f = all(g .≤ 1.0e-8) ? parse(Float64, remaining[1]) : STUPID_MAX
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

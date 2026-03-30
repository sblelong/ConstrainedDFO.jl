using ConstrainedDFO
using JSON
using ProgressBars
using NOMAD

data_path = "/home/sblelong/.julia/dev/ConstrainedDFO/src/benchmark/5-nonlinear/data/mads-pb/"

for (i, problem) in ProgressBar(enumerate(nonlinear_benchmarks[89:end]))
    n = get_dimension(problem)
    x0 = get_x0(problem)
    p = nb_equality_constraints(problem)
    m = nb_inequality_constraints(problem)

    bb(x) = begin
        f = eval_obj(problem, x)
        h = eval_eqs(problem, x)

        if has_inequality_constraints(problem)
            g = eval_ineqs(problem, x)
            bb_outputs = [[f] ; g ; h ; -h]
        else
            bb_outputs = [[f] ; h ; -h]
        end
        return (true, true, bb_outputs)
    end

    noptions = NOMAD.NomadOptions(max_bb_eval = 1000 * n, display_stats = ["BBE", "SOL", "BBO"], display_all_eval = true)
    npb = NomadProblem(
        n,
        1 + 2 * p + m,
        [["OBJ"] ; ["PB" for _ in 1:(2 * p + m)]],
        bb;
        options = noptions
    )

    redirect_to_files(joinpath(data_path, "$(i + 88).txt")) do
        result = solve(npb, x0)
    end
end

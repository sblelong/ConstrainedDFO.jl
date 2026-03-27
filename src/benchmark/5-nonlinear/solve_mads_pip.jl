using ConstrainedDFO
using JSON
using ProgressBars
using ManifoldsBase

data_path = "/home/sblelong/.julia/dev/ConstrainedDFO/src/benchmark/5-nonlinear/data/mads-pip/"
STUPID_MAX = 1.0e20

for (id_instance, problem) in ProgressBar(enumerate(nonlinear_benchmarks))
    n = get_dimension(problem)
    m = nb_inequality_constraints(problem)

    M = get_equality_manifold(problem)
    x0 = get_x0(problem)R
    budget = 1000 * n

    p = nb_equality_constraints(problem)

    open(joinpath(data_path, "param.txt"), "w") do io
        write(io, "DIMENSION $(n)\n")
        write(io, "BB_EXE \"\$julia /home/sblelong/.julia/dev/ConstrainedDFO/src/benchmark/5-nonlinear/blackbox.jl \$$(id_instance)\"\n")

        write(io, "BB_OUTPUT_TYPE OBJ ")
        for i in 1:m
            write(io, "EB ")
        end
        for i in 1:(p - 1)
            write(io, "EQPB ")
        end
        write(io, "EQPB\n")

        write(io, "X0 ( ")
        for i in 1:(n - 1)
            write(io, "$(x0[i]) ")
        end
        write(io, "$(x0[n]) )\n")

        write(io, "MAX_BB_EVAL $(budget)\n")

        write(io, "QUAD_MODEL_SEARCH no\nNM_SEARCH no\nDIRECTION_TYPE ORTHO 2N\nMADSPIP_OPTIMIZATION yes\n")

        write(io, "DISPLAY_DEGREE 0\n")
        write(io, "DISPLAY_ALL_EVAL yes\n")

        history_file_name = joinpath(data_path, "$(id_instance).txt")
        write(io, "STATS_FILE \"$(history_file_name)\" BBE ( SOL ) BBO")
    end

    # Solve the problem with NOMAD
    run(ignorestatus(`/home/sblelong/dev/nomad4dev/build/release/bin/nomad $(joinpath(data_path, "param.txt"))`))
end

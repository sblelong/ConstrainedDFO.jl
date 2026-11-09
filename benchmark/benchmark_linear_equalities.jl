using ConstrainedDFO
using NLPModels, CUTEst
using NOMAD

include(joinpath(@__DIR__, "utils", "nlp_models_utils.jl"))
include(joinpath(@__DIR__, "utils", "solve_nomad.jl"))
include(joinpath(@__DIR__, "utils", "project.jl"))
include(joinpath(@__DIR__, "utils", "problem_selection.jl"))
include(joinpath(@__DIR__, "utils", "progress_table.jl"))

log_path_base = joinpath(@__DIR__, "logs", "linear_equalities")

# Points violating any equality constraint by more than tol_eqs return FAILURE_MAX as objective and inequality constraints.
tol_eqs = 1.0e-8

println("Running benchmark with linear equality constraints...")

problems_names = linear_equality_problems()

table = ProgressTable(["DFRO", "NOMAD"], problems_names)
print_header(table)

for problem_name in problems_names
    start_row(table, problem_name)
    nlp = CUTEstModel(problem_name)
    BI = nlp_to_bb(nlp; tol_eqs = tol_eqs)

    dimension = get_dimension(BI)
    x0f = feasible_first_point(BI)
    if isnothing(x0f)
        finalize(nlp)
        continue
    end
    BI.x0 = x0f

    logs_path = joinpath(log_path_base, "dfro")
    mkpath(logs_path)
    timed_cell(table) do
        redirect_to_files(joinpath(logs_path, "$(problem_name).log")) do
            try
                Random.seed!(SEED)
                result_dfro = DFROSolver(BI; max_evals = 200 * (dimension + 1), display_p0_if_feasible = true)
                true
            catch e
                println("DFRO was unable to solve this problem. See the exception: $(e)")
                false
            end
        end
    end

    logs_path = joinpath(log_path_base, "mads_converter")
    mkpath(logs_path)
    timed_cell(table) do
        redirect_to_files(joinpath(logs_path, "$(problem_name).log")) do
            try
                # Equality constraints as A x = b, in terms of the free variables like the blackbox
                eq_idcs = nlp.meta.jfix
                A_full, c0 = get_linear_part(nlp, eq_idcs)
                A, b = reduce_linear(FixedVarsReduction(nlp), A_full, nlp.meta.ucon[eq_idcs] .- c0)

                Random.seed!(SEED)
                res_nomad_converter = solve_nomad_converter(BI, A, b; converter = :SVD, barrier = :PB, max_evals = 200 * (dimension + 1))
                true
            catch e
                println("NOMAD was unable to solve this problem. See the exception: $(e)")
                false
            end
        end
    end

    finalize(nlp)
    end_row(table)
end
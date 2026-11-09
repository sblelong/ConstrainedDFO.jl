using ConstrainedDFO
using NLPModels, CUTEst
using PRIMA
using NOMAD

include(joinpath(@__DIR__, "utils", "nlp_models_utils.jl"))
include(joinpath(@__DIR__, "utils", "solve_nomad.jl"))
include(joinpath(@__DIR__, "utils", "solve_cobyla.jl"))
include(joinpath(@__DIR__, "utils", "problem_selection.jl"))
include(joinpath(@__DIR__, "utils", "progress_table.jl"))

# Points violating any equality constraint by more than tol_eqs return FAILURE_MAX as objective and inequality constraints.
# Set to nothing for the relaxable setting.
tol_eqs = 1.0e-8

log_path_base = joinpath(@__DIR__, "logs", isnothing(tol_eqs) ? "general_relaxable" : "general_unrelaxable")
log_radii_base = joinpath(@__DIR__, "logs", "dfro_radii", "NOverSqrtSpectral")
dfro_logs_avail = isdir(log_radii_base)

println("Running general benchmark in the $(isnothing(tol_eqs) ? "relaxable" : "unrelaxable") setting.")
dfro_logs_avail && println("DFRO logs detected for these problems: they are reused instead of solving again.")

problems_names = general_problems()

table = ProgressTable(["DFRO", "COBYLA", "MADS-PIP", "MADS-PB"], problems_names)
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

    if dfro_logs_avail && "$(problem_name).log" ∈ readdir(log_radii_base)
        print_cell(table, "reused") # Previous logs for DFRO can be used regardless of relaxable or not, as it is a feasible solver.
    else
        logs_path = joinpath(log_path_base, "dfro")
        mkpath(logs_path)
        timed_cell(table) do
            redirect_to_files(joinpath(logs_path, "$(problem_name).log")) do
                try
                    Random.seed!(SEED)
                    result_dfro = DFROSolver(BI; max_evals = 200 * (dimension + 1), invertibility_bound = NOverSqrtSpectral(), display_p0_if_feasible = true, tol_eqs = tol_eqs)
                    true
                catch e
                    println("DFRO was unable to solve this problem. See the exception: $(e)")
                    false
                end
            end
        end
    end

    logs_path = joinpath(log_path_base, "cobyla")
    mkpath(logs_path)
    timed_cell(table) do
        redirect_to_files(joinpath(logs_path, "$(problem_name).log")) do
            try
                Random.seed!(SEED)
                result_cobyla = solve_cobyla(nlp, x0f; max_evals = 200 * (dimension + 1), tol_eqs = tol_eqs)
                true
            catch e
                println("COBYLA was unable to solve this problem. See the exception: $(e)")
                false
            end
        end
    end

    logs_path = joinpath(log_path_base, "mads-pip")
    mkpath(logs_path)
    timed_cell(table) do
        redirect_to_files(joinpath(logs_path, "$(problem_name).log")) do
            try
                Random.seed!(SEED)
                mads_pip_sol = solve_mads_pip(BI; x0 = x0f, max_evals = 200 * (dimension + 1))
                true
            catch e
                println("MADS-PIP was unable to solve this problem. See the exception: $(e)")
                false
            end
        end
    end

    logs_path = joinpath(log_path_base, "mads-pb")
    mkpath(logs_path)
    timed_cell(table) do
        redirect_to_files(joinpath(logs_path, "$(problem_name).log")) do
            try
                Random.seed!(SEED)
                mads_pb_sol = solve_nomad(BI; x0 = x0f, max_evals = 200 * (dimension + 1), tol_eqs = tol_eqs)
                true
            catch e
                println("MADS-PB was unable to solve this problem. See the exception: $(e)")
                false
            end
        end
    end

    finalize(nlp)
    end_row(table)
end

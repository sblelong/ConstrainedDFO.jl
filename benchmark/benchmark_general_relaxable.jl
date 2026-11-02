using ConstrainedDFO
using NLPModels, CUTEst
using PRIMA
using NOMAD

include(joinpath(@__DIR__, "utils", "nlp_models_utils.jl"))
include(joinpath(@__DIR__, "utils", "solve_nomad.jl"))
include(joinpath(@__DIR__, "utils", "solve_cobyla.jl"))
include(joinpath(@__DIR__, "utils", "problem_selection.jl"))

log_path_base = joinpath(@__DIR__, "logs", "general_relaxable")

println("Running general benchmark in the relaxable setting.")

problems_names = general_problems()

for problem_name in problems_names
    print("$(problem_name)... ")
    nlp = CUTEstModel(problem_name)
    BI = nlp_to_bb(nlp)

    dimension = get_dimension(BI)
    x0f = feasible_first_point(BI)
    if isnothing(x0f)
        finalize(nlp)
        continue
    end
    BI.x0 = x0f

    print("DFRO | ")
    logs_path = joinpath(log_path_base, "dfro")
    mkpath(logs_path)
    redirect_to_files(joinpath(logs_path, "$(problem_name).log")) do
        try
            Random.seed!(SEED)
            result_dfro = DFROSolver(BI; max_evals = 200 * (dimension + 1), invertibility_bound = NOverSqrtSpectral(), display_p0_if_feasible = true)
        catch e
            println("DFRO was unable to solve this problem. See the exception: $(e)")
        end
    end

    print("COBYLA | ")
    logs_path = joinpath(log_path_base, "cobyla")
    mkpath(logs_path)
    redirect_to_files(joinpath(logs_path, "$(problem_name).log")) do
        try
            Random.seed!(SEED)
            result_cobyla = solve_cobyla(nlp, x0f; max_evals = 200 * (dimension + 1))
        catch e
            println("COBYLA was unable to solve this problem. See the exception: $(e)")
        end
    end

    print("MADS-PIP | ")
    logs_path = joinpath(log_path_base, "mads-pip")
    mkpath(logs_path)
    redirect_to_files(joinpath(logs_path, "$(problem_name).log")) do
        try
            Random.seed!(SEED)
            mads_pip_sol = solve_mads_pip(BI; x0 = x0f, max_evals = 200 * (dimension + 1))
        catch e
            println("MADS-PIP was unable to solve this problem. See the exception: $(e)")
        end
    end

    print("MADS-PB | ")
    logs_path = joinpath(log_path_base, "mads-pb")
    mkpath(logs_path)
    redirect_to_files(joinpath(logs_path, "$(problem_name).log")) do
        try
            Random.seed!(SEED)
            mads_pb_sol = solve_nomad(BI; x0 = x0f, max_evals = 200 * (dimension + 1))
        catch e
            println("MADS-PB was unable to solve this problem. See the exception: $(e)")
        end
    end
    println("✓")
    finalize(nlp)
end

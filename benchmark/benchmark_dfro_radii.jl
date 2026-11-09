using ConstrainedDFO
using NLPModels, CUTEst

include(joinpath(@__DIR__, "utils", "nlp_models_utils.jl"))
include(joinpath(@__DIR__, "utils", "project.jl"))
include(joinpath(@__DIR__, "utils", "problem_selection.jl"))
include(joinpath(@__DIR__, "utils", "progress_table.jl"))

log_path_base = joinpath(@__DIR__, "logs", "dfro_radii")

println("Running DFRO invertibility radii benchmark.")

problems_names = general_problems()

table = ProgressTable(["OneOverSpectral", "OneOverSqrtSpectral", "NOverSpectral", "NOverSqrtSpectral"], problems_names)
print_header(table)

for problem_name in problems_names
    start_row(table, problem_name)
    nlp = CUTEstModel(problem_name)
    BI = nlp_to_bb(nlp)

    dimension = get_dimension(BI)
    x0f = feasible_first_point(BI)
    if isnothing(x0f)
        finalize(nlp)
        continue
    end
    BI.x0 = x0f

    logs_path = joinpath(log_path_base, "OneOverSpectral")
    mkpath(logs_path)
    timed_cell(table) do
        redirect_to_files(joinpath(logs_path, "$(problem_name).log")) do
            try
                Random.seed!(SEED)
                solution = DFROSolver(BI; max_evals = 200 * (dimension + 1), invertibility_bound = OneOverSpectral(), display_p0_if_feasible = true)
                true
            catch e
                println("DFROSolver was unable to solve this problem. See the exception: $(e)")
                false
            end
        end
    end

    logs_path = joinpath(log_path_base, "OneOverSqrtSpectral")
    mkpath(logs_path)
    timed_cell(table) do
        redirect_to_files(joinpath(logs_path, "$(problem_name).log")) do
            try
                Random.seed!(SEED)
                solution = DFROSolver(BI; max_evals = 200 * (dimension + 1), invertibility_bound = OneOverSqrtSpectral(), display_p0_if_feasible = true)
                true
            catch e
                println("DFROSolver was unable to solve this problem. See the exception: $(e)")
                false
            end
        end
    end

    logs_path = joinpath(log_path_base, "NOverSpectral")
    mkpath(logs_path)
    timed_cell(table) do
        redirect_to_files(joinpath(logs_path, "$(problem_name).log")) do
            try
                Random.seed!(SEED)
                solution = DFROSolver(BI; max_evals = 200 * (dimension + 1), invertibility_bound = NOverSpectral(), display_p0_if_feasible = true)
                true
            catch e
                println("DFROSolver was unable to solve this problem. See the exception: $(e)")
                false
            end
        end
    end

    logs_path = joinpath(log_path_base, "NOverSqrtSpectral")
    mkpath(logs_path)
    timed_cell(table) do
        redirect_to_files(joinpath(logs_path, "$(problem_name).log")) do
            try
                Random.seed!(SEED)
                solution = DFROSolver(BI; max_evals = 200 * (dimension + 1), invertibility_bound = NOverSqrtSpectral(), display_p0_if_feasible = true)
                true
            catch e
                println("DFROSolver was unable to solve this problem. See the exception: $(e)")
                false
            end
        end
    end

    finalize(nlp)
    end_row(table)
end

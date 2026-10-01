using ConstrainedDFO
using NLPModels, CUTEst

include(joinpath(@__DIR__, "utils", "nlp_models_utils.jl"))

log_path_base = joinpath(@__DIR__, "logs", "dfro_radii")

println("Running DFRO invertibility radii benchmark.")

# Only nonlinear equality constraints and no bounds
problems_names = CUTEst.select_sif_problems(
    max_var = 100,
    min_con = 1,
    only_equ_con = true,
    contype = [6, 7],
    custom_filter = meta -> meta["variables"]["number"] > meta["constraints"]["number"]
)
# The following problems won't work with DFRO:
exclude_from_dfro = [
    "ALLINITC", # can't project the first guess correctly
    "S316-322", # also a Jacobian rank problem
    "HS61", # Jacobian rank problem
]
filter!(e -> e ∉ exclude_from_dfro, problems_names)

println("Solving with OneOverSpectral...")
for problem_name in problems_names
    print("$(problem_name)... ")
    nlp = CUTEstModel(problem_name)
    BI = nlp_to_bb(nlp)

    dimension = get_dimension(BI)

    logs_path = joinpath(log_path_base, "OneOverSpectral")
    mkpath(logs_path)
    redirect_to_files(joinpath(logs_path, "$(problem_name).log")) do
        try
            res_dfro = DFROSolver(BI; max_evals = 1000 * (dimension + 1), invertibility_bound = OneOverSpectral())
        catch e
            println("DFROSolver was unable to solve this problem. See the exception: $(e)")
        end
    end
    finalize(nlp)
    println("✓")
end

println("Solving with OneOverSqrtSpectral...")
for problem_name in problems_names
    print("$(problem_name)... ")
    nlp = CUTEstModel(problem_name)
    BI = nlp_to_bb(nlp)

    dimension = get_dimension(BI)

    logs_path = joinpath(log_path_base, "OneOverSqrtSpectral")
    mkpath(logs_path)
    redirect_to_files(joinpath(logs_path, "$(problem_name).log")) do
        try
            res_dfro = DFROSolver(BI; max_evals = 1000 * (dimension + 1), invertibility_bound = OneOverSqrtSpectral())
        catch e
            println("DFROSolver was unable to solve this problem. See the exception: $(e)")
        end
    end
    finalize(nlp)
    println("✓")
end

println("Solving with NOverSpectral...")
for problem_name in problems_names
    print("$(problem_name)... ")
    nlp = CUTEstModel(problem_name)
    BI = nlp_to_bb(nlp)

    dimension = get_dimension(BI)

    logs_path = joinpath(log_path_base, "NOverSpectral")
    mkpath(logs_path)
    redirect_to_files(joinpath(logs_path, "$(problem_name).log")) do
        try
            res_dfro = DFROSolver(BI; max_evals = 1000 * (dimension + 1), invertibility_bound = NOverSpectral())
        catch e
            println("DFROSolver was unable to solve this problem. See the exception: $(e)")
        end
    end
    finalize(nlp)
    println("✓")
end

println("Solving with NOverSqrtSpectral...")
for problem_name in problems_names
    print("$(problem_name)... ")
    nlp = CUTEstModel(problem_name)
    BI = nlp_to_bb(nlp)

    dimension = get_dimension(BI)

    logs_path = joinpath(log_path_base, "NOverSqrtSpectral")
    mkpath(logs_path)
    redirect_to_files(joinpath(logs_path, "$(problem_name).log")) do
        try
            res_dfro = DFROSolver(BI; max_evals = 1000 * (dimension + 1), invertibility_bound = NOverSqrtSpectral())
        catch e
            println("DFROSolver was unable to solve this problem. See the exception: $(e)")
        end
    end
    finalize(nlp)
    println("✓")
end

using ConstrainedDFO
using NLPModels, CUTEst

include(joinpath(@__DIR__, "utils", "nlp_models_utils.jl"))
include(joinpath(@__DIR__, "utils", "project.jl"))

log_path_base = joinpath(@__DIR__, "logs", "dfro_radii")

println("Running DFRO invertibility radii benchmark.")

CUTEst.set_mastsif()

# Only nonlinear equality constraints, inequality and bounds possible.
filter(meta) = (meta["variables"]["number"] > meta["constraints"]["number"]) && (meta["constraints"]["equality"] > 0)
problems_names = CUTEst.select_sif_problems(
    max_var = 100,
    min_con = 1,
    contype = [6, 7],
    custom_filter = filter
)
# The following problems won't work with DFRO:
to_exclude = [
    # Can't find a feasible first guess
    "SSINE",
    "VANDANIUMS",

    # Can't find a feasible solution in the first subproblem
    # "AIRCRFTA",
    # "ALLINITC",
    # "CONCON",
    # "CORE1",
    # "DECONVBE",
    # "DISC2",
    # "DNIEPER",
    # "ERRINBAR",

    # Can't compute invertibility radius: Hessian contains Inf or NaN
    "BA-L1",
    "CSFI1",
    "CSFI2",
    "LAKES",
    "STRTCHDVNE",
]
filter!(e -> e ∉ to_exclude, problems_names)

for problem_name in problems_names
    print("$(problem_name)... ")
    nlp = CUTEstModel(problem_name)
    BI = nlp_to_bb(nlp)

    dimension = get_dimension(BI)
    try
        x0f = make_x0_feasible(BI)
        BI.x0 = x0f
    catch e
        println("Couldn't find a feasible first point. Skipping.")
        continue
    end

    print("OneOverSpectral ")
    logs_path = joinpath(log_path_base, "OneOverSpectral")
    mkpath(logs_path)
    redirect_to_files(joinpath(logs_path, "$(problem_name).log")) do
        try
            solution = DFROSolver(BI; max_evals = 200 * (dimension + 1), invertibility_bound = OneOverSpectral())
        catch e
            println("DFROSolver was unable to solve this problem. See the exception: $(e)")
        end
    end

    print("OneOverSqrtSpectral ")
    logs_path = joinpath(log_path_base, "OneOverSqrtSpectral")
    mkpath(logs_path)
    redirect_to_files(joinpath(logs_path, "$(problem_name).log")) do
        try
            solution = DFROSolver(BI; max_evals = 200 * (dimension + 1), invertibility_bound = OneOverSqrtSpectral())
        catch e
            println("DFROSolver was unable to solve this problem. See the exception: $(e)")
        end
    end

    print("NOverSpectral ")
    logs_path = joinpath(log_path_base, "NOverSpectral")
    mkpath(logs_path)
    redirect_to_files(joinpath(logs_path, "$(problem_name).log")) do
        try
            solution = DFROSolver(BI; max_evals = 200 * (dimension + 1), invertibility_bound = NOverSpectral())
        catch e
            println("DFROSolver was unable to solve this problem. See the exception: $(e)")
        end
    end
    print("NOverSqrtSpectral ")
    logs_path = joinpath(log_path_base, "NOverSqrtSpectral")
    mkpath(logs_path)
    redirect_to_files(joinpath(logs_path, "$(problem_name).log")) do
        try
            solution = DFROSolver(BI; max_evals = 200 * (dimension + 1), invertibility_bound = NOverSqrtSpectral())
        catch e
            println("DFROSolver was unable to solve this problem. See the exception: $(e)")
        end
    end

    finalize(nlp)
    println("✓")
end

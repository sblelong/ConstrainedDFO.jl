using ConstrainedDFO
using NLPModels, CUTEst
using PRIMA
using NOMAD

include(joinpath(@__DIR__, "utils", "nlp_models_utils.jl"))
include(joinpath(@__DIR__, "utils", "solve_nomad.jl"))
include(joinpath(@__DIR__, "utils", "solve_cobyla.jl"))

log_path_base = joinpath(@__DIR__, "logs", "general_unrelaxable")

# Points violating any equality constraint by more than tol_eqs return FAILURE_MAX as objective and inequality constraints.
tol_eqs = 1.0e-8

println("Running general benchmark in the unrelaxable setting.")

CUTEst.set_mastsif()
# Only nonlinear equality constraints, inequality and bounds possible.
filter(meta) = (meta["variables"]["number"] > meta["constraints"]["number"]) && (meta["constraints"]["equality"] > 0)
problems_names = CUTEst.select_sif_problems(
    max_var = 40,
    min_con = 1,
    contype = [6, 7],
    custom_filter = filter
)
# The following problems won't work with DFRO:
to_exclude = [
    # Manifold dimension is 0 after handling fixed variables
    "AIRCRFTA",
    "TRIGGER",

    # Can't find a feasible first guess
    "SSINE",
    "VANDANIUMS",

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
    BI = nlp_to_bb(nlp; tol_eqs = tol_eqs)

    dimension = get_dimension(BI)
    x0f::Vector{Float64} = zeros(dimension)
    try
        x0f = make_x0_feasible(BI)
        BI.x0 = x0f
    catch e
        println("Couldn't find a feasible first point. Skipping.")
        continue
    end
    # Otherwise, the first evaluation of every solver would fail.
    if any(abs.(eval_eqs(BI, x0f)) .> tol_eqs)
        println("The first point violates the equality constraints by more than $(tol_eqs). Skipping.")
        continue
    end

    print("DFRO | ")
    logs_path = joinpath(log_path_base, "dfro")
    mkpath(logs_path)
    redirect_to_files(joinpath(logs_path, "$(problem_name).log")) do
        try
            result_dfro = DFROSolver(BI; max_evals = 200 * (dimension + 1), invertibility_bound = NOverSqrtSpectral(), display_p0_if_feasible = true, tol_eqs = tol_eqs)
        catch e
            println("DFRO was unable to solve this problem. See the exception: $(e)")
        end
    end

    print("COBYLA | ")
    logs_path = joinpath(log_path_base, "cobyla")
    mkpath(logs_path)
    redirect_to_files(joinpath(logs_path, "$(problem_name).log")) do
        try
            result_cobyla = solve_cobyla(nlp, x0f; max_evals = 200 * (dimension + 1), tol_eqs = tol_eqs)
        catch e
            println("COBYLA was unable to solve this problem. See the exception: $(e)")
        end
    end

    print("MADS-PIP | ")
    logs_path = joinpath(log_path_base, "mads-pip")
    mkpath(logs_path)
    redirect_to_files(joinpath(logs_path, "$(problem_name).log")) do
        try
            mads_pip_sol = solve_mads_pip(BI; x0 = x0f, max_evals = 200 * (dimension + 1), tol_eqs = tol_eqs)
        catch e
            println("MADS-PIP was unable to solve this problem. See the exception: $(e)")
        end
    end

    print("MADS-PB ")
    logs_path = joinpath(log_path_base, "mads-pb")
    mkpath(logs_path)
    redirect_to_files(joinpath(logs_path, "$(problem_name).log")) do
        try
            mads_pb_sol = solve_nomad(BI; x0 = x0f, max_evals = 200 * (dimension + 1), tol_eqs = tol_eqs)
        catch e
            println("MADS-PB was unable to solve this problem. See the exception: $(e)")
        end
    end
    println("✓")
end

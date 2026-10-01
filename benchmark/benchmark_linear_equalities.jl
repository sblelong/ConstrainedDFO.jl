using ConstrainedDFO
using NLPModels, CUTEst
using NOMAD

include(joinpath(@__DIR__, "utils", "nlp_models_utils.jl"))
include(joinpath(@__DIR__, "utils", "solve_nomad.jl"))

log_path_base = joinpath(@__DIR__, "logs", "linear_equalities")

println("Running benchmark with linear equality constraints...")
println()

CUTEst.set_mastsif()

# Select SIF problems: has at least one equality constraint
filter(meta) = meta["constraints"]["equality"] > 0 && meta["variables"]["number"] > meta["constraints"]["number"]
problems_names = CUTEst.select_sif_problems(
    max_var = 100,
    custom_filter = filter
)
# NLP filter: the equality constraints have to all be linear (nlp.meta.lin == nlp.meta.jfix).
function only_linear_equalities(name::String)
    nlp = CUTEstModel(name)
    result = nlp.meta.lin == nlp.meta.jfix
    finalize(nlp)
    return result
end
filter!(only_linear_equalities, problems_names)

# The following problems lead to bugs with either of the two solvers that are hard to solve.
to_exclude = [
    "LSNNODOC", # the first guess has a Jacobian with wrong rank (doesn't mean the dim(M)=n-p requirement)
    "DEGENLPA", # NOMAD fails on this problem.
    "DEGENLPB", # NOMAD tweaks the bounds and ends up having lb[2] ≥ ub[2]
    "HS32", # The only one to have an inequality constraint, might as well only consider problems with bounds at most.
    "DALLASS", # Jacobian rank issue
    "HIMMELBJ", # Cannot compute feasible first guess.
    "LINSPANH",
    "NASH",
    "SPANHYD",
    "WATER",
]
filter!(e -> e ∉ to_exclude, problems_names)

println("Solving with DFRO...")
for problem_name in problems_names
    print("$(problem_name)... ")
    nlp = CUTEstModel(problem_name)
    BI = nlp_to_bb(nlp)

    dimension = get_dimension(BI)

    logs_path = joinpath(log_path_base, "dfro")
    mkpath(logs_path)
    redirect_to_files(joinpath(logs_path, "$(problem_name).log")) do
        try
            BI.x0 = make_x0_feasible(nlp)
            res_dfro = DFROSolver(BI; max_evals = 1000 * (dimension + 1), display_p0_if_feasible = true)
        catch e
            println("DFROSolver was unable to solve this problem. See the exception: $(e)")
        end
    end
    finalize(nlp)
    println("✓")
end

println()

println("Solving with MADS and converters...")
for problem_name in problems_names
    print("$(problem_name)... ")
    nlp = CUTEstModel(problem_name)
    BI = nlp_to_bb(nlp)

    dimension = get_dimension(BI)

    x0 = ConstrainedDFO.get_x0(BI)
    eq_idcs = nlp.meta.jfix
    A = Matrix{Float64}(jac(nlp, x0))[eq_idcs, :]
    h0 = cons(nlp, zeros(dimension))
    b = -h0[eq_idcs]

    logs_path = joinpath(log_path_base, "mads_converter")
    mkpath(logs_path)
    try
        # Make the first guess feasible for equalities and bounds
        redirect_to_files(joinpath(logs_path, "$(problem_name).log")) do
            BI.x0 = make_x0_feasible(nlp)
            res_nomad_converter = solve_nomad_converter(BI, A, b; converter = :SVD, barrier = :PB, max_evals = 1000 * (dimension + 1))
        end
    catch e
        println("MADS solver was unable to solve this problem. See the exception: $(e)")
    end
    finalize(nlp)
    println("✓")
end

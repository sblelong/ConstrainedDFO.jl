using ConstrainedDFO
using NLPModels, CUTEst
using NOMAD

include(joinpath(@__DIR__, "utils", "nlp_models_utils.jl"))
include(joinpath(@__DIR__, "utils", "solve_nomad.jl"))

log_path_base = joinpath(@__DIR__, "logs", "linear_equalities")

println("Running benchmark with linear equality constraints...")
println()

CUTEst.set_mastsif()

# Small problems with linear equality constraints. Inequality constraints and bounds are allowed.
# Select SIF problems: has at least one equality constraint
filter(meta) = meta["constraints"]["equality"] > 0 && meta["variables"]["number"] > meta["constraints"]["number"]
# Small problems with linear equality constraints. Inequality constraints and bounds are allowed.
problems_names = CUTEst.select_sif_problems(
    max_var = 10,
    custom_filter = filter
)

# NLP filter: the equality constraints have to all be linear (nlp.meta.lin == nlp.meta.jfix).
function has_linear_equalities(name::String)
    nlp = CUTEstModel(name)
    result = nlp.meta.lin == nlp.meta.jfix
    finalize(nlp)
    return result
end
filter!(has_linear_equalities, problems_names)

# The following problems won't work with DFRO:
exclude_from_dfro = [
    "ALLINITC", # can't project the first guess correctly
    "LSNNODOC", # the first guess has a Jacobian with wrong rank (doesn't mean the dim(M)=n-p requirement)
    "S316-322", # also a Jacobian rank problem
    "HS61", # Jacobian rank problem
    "BT13", # TODO put this one back, it's just too long to solve but it works
    "HS107", # TODO put it back, it's too long.
]
filter!(e -> e ∉ exclude_from_dfro, problems_names)

problems_names = problems_names[1:20]

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
            res_dfro = DFROSolver(BI; max_evals = 1000 * (dimension + 1), display_first_infeasible = false)
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
    redirect_to_files(joinpath(logs_path, "$(problem_name).log")) do
        res_nomad_converter = solve_nomad_converter(BI, A, b; converter = :SVD, barrier = :PB, max_evals = 1000 * (dimension + 1))
    end
    finalize(nlp)
    println("✓")
end

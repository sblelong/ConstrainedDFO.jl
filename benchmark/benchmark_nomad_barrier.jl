using ConstrainedDFO
using NLPModels, CUTEst
using NOMAD

include(joinpath(@__DIR__, "utils", "nlp_models_utils.jl"))
include(joinpath(@__DIR__, "utils", "solve_nomad.jl"))

log_path_base = joinpath(@__DIR__, "logs", "nomad_barrier")

# Small problems, only equality constraints and no bounds
problems_names = CUTEst.select_sif_problems(
    max_var = 10,
    min_con = 1,
    only_equ_con = true,
    custom_filter = meta -> meta["variables"]["number"] > meta["constraints"]["number"]
)

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
    mkpath(dirname(logs_path))
    redirect_to_files(joinpath(logs_path, "$(problem_name).log")) do
        try
            res_dfro = DFROSolver(BI; max_evals = 1000 * (dimension + 1))
        catch e
            println("DFROSolver was unable to solve this problem. See the exception: $(e)")
        end
    end
    finalize(nlp)
    println("✓")
end

# println()

# println("Solving with MADS-EB...")
# for problem_name in problems_names
#     print("$(problem_name)... ")
#     nlp = CUTEstModel(problem_name)
#     BI = nlp_to_bb(nlp)

#     dimension = get_dimension(BI)

#     logs_path = joinpath(log_path_base, "mads_eb")
#     mkpath(dirname(logs_path))
#     redirect_to_files(joinpath(logs_path, "$(problem_name).log")) do
#         res_nomad_eb = solve_nomad(BI; barrier = :EB, max_evals = 1000 * (dimension + 1))
#     end
#     finalize(nlp)
#     println("✓")
# end

# println()

# println("Solving with MADS-PB...")
# for problem_name in problems_names
#     print("$(problem_name)... ")
#     nlp = CUTEstModel(problem_name)
#     BI = nlp_to_bb(nlp)

#     dimension = get_dimension(BI)

#     logs_path = joinpath(log_path_base, "mads_pb")
#     mkpath(dirname(logs_path))
#     redirect_to_files(joinpath(logs_path, "$(problem_name).log")) do
#         res_nomad_eb = solve_nomad(BI; barrier = :PB, max_evals = 1000 * (dimension + 1))
#     end
#     finalize(nlp)
#     println("✓")
# end

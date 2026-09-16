using ConstrainedDFO
using NLPModels, CUTEst
using NOMAD

include(joinpath(@__DIR__, "utils", "nlp_models_utils.jl"))
include(joinpath(@__DIR__, "utils", "solve_nomad.jl"))

log_path_base = joinpath(@__DIR__, "data", "nomad_barrier")

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
]
problems_names_dfro = setdiff(Set(problems_names), Set(exclude_from_dfro))

# println("Solving with DFRO...")
# for problem_name in problems_names_dfro
#     print("$(problem_name)... ")
#     nlp = CUTEstModel(problem_name)
#     BI = nlp_to_bb(nlp)

#     dimension = get_dimension(BI)

#     redirect_to_files(joinpath(log_path_base, "dfro", "$(problem_name).log")) do
#         try
#             res_dfro = DFROSolver(BI; max_evals = 1000 * (dimension + 1))
#         catch e
#             println("DFROSolver was unable to solve this problem. See the exception: $(e)")
#         end
#     end
#     finalize(nlp)
#     println("✓")
# end

# println()

# println("Solving with NOMAD-EB...")
# for problem_name in problems_names_dfro
#     print("$(problem_name)... ")
#     nlp = CUTEstModel(problem_name)
#     BI = nlp_to_bb(nlp)

#     dimension = get_dimension(BI)

#     redirect_to_files(joinpath(log_path_base, "nomad-eb", "$(problem_name).log")) do
#         res_nomad_eb = solve_nomad(BI; barrier = :EB, max_evals = 1000 * (dimension + 1))
#     end
#     finalize(nlp)
#     println("✓")
# end

println()

println("Solving with NOMAD-PB...")
for problem_name in problems_names_dfro
    print("$(problem_name)... ")
    nlp = CUTEstModel(problem_name)
    BI = nlp_to_bb(nlp)

    dimension = get_dimension(BI)

    redirect_to_files(joinpath(log_path_base, "nomad-pb", "$(problem_name).log")) do
        res_nomad_eb = solve_nomad(BI; barrier = :PB, max_evals = 1000 * (dimension + 1))
    end
    finalize(nlp)
    println("✓")
end

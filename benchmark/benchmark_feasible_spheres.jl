using ConstrainedDFO
using NLPModels, CUTEst
using ManifoldsBase, Manifolds, Manopt

include(joinpath(@__DIR__, "utils", "nlp_models_utils.jl"))
include(joinpath(@__DIR__, "utils", "solve_manopt.jl"))

log_path_base = joinpath(@__DIR__, "logs", "feasible_spheres")

println("Running benchmark to compare feasible solvers on sphere-constrained problems.")
println()

CUTEst.set_mastsif()

# Small unconstrained problems, think of giving feasible first guesses (just take x0/||x0||).
problems_names = CUTEst.select_sif_problems(
    max_var = 10,
    contype = :unc
)

# The following problems won't work with DFRO:
exclude_from_dfro = []
filter!(e -> e ∉ exclude_from_dfro, problems_names)

problems_names = problems_names[1:20]

println("Solving with DFRO (Projection Retraction)...")
for problem_name in problems_names
    print("$(problem_name)... ")

    nlp = CUTEstModel(problem_name)
    BI = nlp_to_bb(nlp)
    dimension = get_dimension(BI)

    h(x) = [sum(x .^ 2) - 1]
    M = EqualityManifold(h, dimension - 1, dimension)

    f(x) = eval_objective(BI, x)
    x0 = ConstrainedDFO.get_x0(BI)
    p0 = norm(x0) == 0 ? [[1.0] ; [0.0 for _ in 1:(dimension - 1)]] : x0 ./ norm(x0)

    logs_path = joinpath(log_path_base, "dfro-projection")
    mkpath(logs_path)
    redirect_to_files(joinpath(logs_path, "$(problem_name).log")) do
        try
            res_dfro = DFROSolver(M, f, p0; max_evals = 1000 * (dimension + 1), invertibility_bound = NOverSqrtSpectral())
        catch e
            println("DFROSolver was unable to solve this problem. See the exception: $(e)")
        end
    end
    finalize(nlp)
    println("✓")
end

println("")

println("Solving with DFRO (Exponential Retraction)...")
for problem_name in problems_names
    print("$(problem_name)... ")

    nlp = CUTEstModel(problem_name)
    BI = nlp_to_bb(nlp)
    dimension = get_dimension(BI)

    M = Manifolds.Sphere(dimension - 1)

    f(x) = eval_objective(BI, x)
    x0 = ConstrainedDFO.get_x0(BI)
    p0 = norm(x0) == 0 ? [[1.0] ; [0.0 for _ in 1:(dimension - 1)]] : x0 ./ norm(x0)

    logs_path = joinpath(log_path_base, "dfro-exp")
    mkpath(logs_path)
    redirect_to_files(joinpath(logs_path, "$(problem_name).log")) do
        try
            res_dfro = DFROSolver(M, f, p0; max_evals = 1000 * (dimension + 1), retraction_method = ExponentialRetraction(), invertibility_bound = ExactInvertibility())
        catch e
            println("DFROSolver was unable to solve this problem. See the exception: $(e)")
        end
    end
    finalize(nlp)
    println("✓")
end

println("")

println("Solving with Manopt.jl...")
for problem_name in problems_names
    print("$(problem_name)... ")
    nlp = CUTEstModel(problem_name)
    BI = nlp_to_bb(nlp)

    dimension = get_dimension(BI)

    logs_path = joinpath(log_path_base, "manopt")
    mkpath(logs_path)
    redirect_to_files(joinpath(logs_path, "$(problem_name).log")) do
        res_manopt = solve_sphere_manopt(BI; max_evals = 1000 * (dimension + 1))
    end
    finalize(nlp)
    println("✓")
end

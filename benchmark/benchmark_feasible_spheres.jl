using ConstrainedDFO
using NLPModels, CUTEst
using ManifoldsBase, Manifolds, Manopt

include(joinpath(@__DIR__, "utils", "nlp_models_utils.jl"))
include(joinpath(@__DIR__, "utils", "solve_manopt.jl"))
include(joinpath(@__DIR__, "utils", "problem_selection.jl"))

log_path_base = joinpath(@__DIR__, "logs", "feasible_spheres")

println("Running benchmark to compare feasible solvers on sphere-constrained problems.")

problems_names = unconstrained_problems()

for problem_name in problems_names
    print("$(problem_name)... ")
    nlp = CUTEstModel(problem_name)
    BI = nlp_to_bb(nlp)

    dimension = get_dimension(BI)

    f(x) = eval_objective(BI, x)
    x0 = ConstrainedDFO.get_x0(BI)
    p0 = norm(x0) == 0 ? [[1.0] ; [0.0 for _ in 1:(dimension - 1)]] : x0 ./ norm(x0)

    print("DFRO-proj | ")
    logs_path = joinpath(log_path_base, "dfro-projection")
    mkpath(logs_path)
    h(x) = [sum(x .^ 2) - 1]
    M = EqualityManifold(h, dimension - 1, dimension)
    redirect_to_files(joinpath(logs_path, "$(problem_name).log")) do
        try
            Random.seed!(SEED)
            solution = DFROSolver(M, f, p0; max_evals = 200 * (dimension + 1), invertibility_bound = NOverSqrtSpectral(), display_p0_if_feasible = true)
        catch e
            println("DFROSolver was unable to solve this problem. See the exception: $(e)")
        end
    end

    print("DFRO-Exp | ")
    logs_path = joinpath(log_path_base, "dfro-exp")
    mkpath(logs_path)
    M = Manifolds.Sphere(dimension - 1)
    redirect_to_files(joinpath(logs_path, "$(problem_name).log")) do
        try
            Random.seed!(SEED)
            solution = DFROSolver(M, f, p0; max_evals = 200 * (dimension + 1), retraction_method = ExponentialRetraction(), invertibility_bound = ExactInvertibility(), display_p0_if_feasible = true)
        catch e
            println("DFROSolver was unable to solve this problem. See the exception: $(e)")
        end
    end

    print("Manopt | ")
    logs_path = joinpath(log_path_base, "manopt")
    mkpath(logs_path)
    redirect_to_files(joinpath(logs_path, "$(problem_name).log")) do
        Random.seed!(SEED)
        res_manopt = solve_sphere_manopt(BI; max_evals = 200 * (dimension + 1))
    end

    finalize(nlp)
    println("✓")

end

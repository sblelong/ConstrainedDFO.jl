using DelimitedFiles

abstract type AbstractBenchmarkProblem end

mutable struct BenchmarkEqualityProblem <: AbstractBenchmarkProblem
    dimension::Int
    objective::Function
    ineq_constraints::Function
    eq_constraints::Function
    x0::Vector{Float64}
    A::Union{Nothing, Matrix{Float64}}
    b::Union{Nothing, Vector{Float64}}
end

function BenchmarkEqualityProblem(dimension::Int, objective::Function, ineq_constraints::Function, eq_constraints::Function, x0; A::Union{Nothing, Matrix{Float64}} = nothing, b::Union{Nothing, Vector{Float64}} = nothing)
    return BenchmarkEqualityProblem(dimension, objective, ineq_constraints, eq_constraints, x0, A, b)
end

get_dimension(BPE::BenchmarkEqualityProblem) = BPE.dimension
eval_obj(BP::AbstractBenchmarkProblem, x) = BP.objective(x)
eval_ineqs(BP::AbstractBenchmarkProblem, x) = BP.ineq_constraints(x)
eval_eqs(BPE::BenchmarkEqualityProblem, x) = BPE.eq_constraints(x)
get_x0(BP::AbstractBenchmarkProblem) = BP.x0
nb_inequality_constraints(BP::AbstractBenchmarkProblem) = length(eval_ineqs(BP, get_x0(BP)))
has_inequality_constraints(BP::AbstractBenchmarkProblem) = nb_inequality_constraints(BP) > 0
nb_equality_constraints(BPE::BenchmarkEqualityProblem) = length(eval_eqs(BPE, get_x0(BPE)))

# Equality constraints
eq_sphere(x; radius::Float64 = 1.0) = sum(x .^ 2) - radius^2

# Objective functions
obj_rosenbrock(x) = sum([100(x[i + 1] - x[i]^2)^2 + (1 - x[i])^2 for i in 1:(length(x) - 1)])

obj_hs6(x) = (1 - x[1])^2
obj_hs7(x) = log(1 + x[1]^2) - x[2]
obj_hs26(x) = (x[1] - x[2])^2 + (x[2] - x[3])^2
obj_hs27(x) = 0.01(x[1] - 1)^2 + (x[2] - x[1]^2)^2
obj_hs47(x) = (x[1] - x[2])^2 + (x[2] - x[3])^3 + (x[3] - x[4])^4 + (x[4] - x[5])^4
obj_hs60(x) = (x[1] - 1)^2 + (x[1] - x[2])^2 + (x[2] - x[3])^4
obj_hs61(x) = 4x[1]^2 + 2x[2]^2 + 2x[3]^2 - 33x[1] + 16x[2] - 24x[3]
obj_hs63(x) = 1000 - x[1]^2 - 2x[2]^2 - x[3]^2 - x[1] * x[2] - x[1] * x[3]
obj_hs71(x) = x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3]
obj_hs74(x) = 3x[1] + 1.0e-6 * x[1]^3 + 2x[2] + 2 / 3 * 1.0e-6 * x[2]^3
obj_hs77(x) = (x[1] - 1)^2 + (x[1] - x[2])^2 + (x[3] - 1)^2 + (x[4] - 1)^4 + (x[5] - 1)^6

c_hs111 = [-6.089, -17.164, -34.054, -5.914, -24.721, -14.986, -24.1, -10.708, -26.662, -22.179]
obj_hs111(x) = begin
    f = 1.0e20
    try
        f = sum([exp(x[i]) * (c_hs111[i] + x[i] - log(sum([exp(x[k]) for k in 1:10]))) for i in 1:10])
    catch e end
    return f
end

eq_hs111(x) = [exp(x[1]) + 2 * exp(x[2]) + 2 * exp(x[3]) + exp(x[6]) + exp(x[10]) - 2, exp(x[4]) + 2 * exp(x[5]) + exp(x[6]) + exp(x[7]) - 1, exp(x[3]) + exp(x[7]) + exp(x[8]) + 2 * exp(x[9]) + exp(x[10]) - 1]

obj_hs81(x) = exp(prod(x)) - 0.5 * (x[1]^3 + x[2]^3 + 1)^2
eq_hs81(x) = [eq_sphere(x; radius = sqrt(10)), x[2] * x[3] - 5x[4] * x[5], x[1]^3 + x[2]^3 + 1]

eq_hs80(x) = [eq_sphere(x; radius = sqrt(10)), x[2] * x[3] - 5x[4] * x[5], x[1]^3 + x[2]^3 + 1]


nonlinear_benchmarks = [
    # HS6
    BenchmarkEqualityProblem(2, obj_hs6, x -> [], x -> [10(x[2] - x[1]^2)], [-1.2, 1.0]),
    BenchmarkEqualityProblem(2, obj_hs6, x -> [], x -> [10(x[2] - x[1]^2)], [4.57, 37.18]),
    BenchmarkEqualityProblem(2, obj_hs6, x -> [], x -> [10(x[2] - x[1]^2)], [45.52, 9.16]),
    BenchmarkEqualityProblem(2, obj_hs6, x -> [], x -> [10(x[2] - x[1]^2)], [-44.02, 22.46]),
    BenchmarkEqualityProblem(2, obj_hs6, x -> [], x -> [10(x[2] - x[1]^2)], [-4.66, 46.8]),

    # HS7
    BenchmarkEqualityProblem(2, obj_hs7, x -> [], x -> [(1 + x[1]^2)^2 + x[2]^2 - 4], [2.0, 2.0]),
    BenchmarkEqualityProblem(2, obj_hs7, x -> [], x -> [(1 + x[1]^2)^2 + x[2]^2 - 4], [34.97, -37.59]),
    BenchmarkEqualityProblem(2, obj_hs7, x -> [], x -> [(1 + x[1]^2)^2 + x[2]^2 - 4], [7.98, -5.98]),
    BenchmarkEqualityProblem(2, obj_hs7, x -> [], x -> [(1 + x[1]^2)^2 + x[2]^2 - 4], [-6.9, -3.82]),
    BenchmarkEqualityProblem(2, obj_hs7, x -> [], x -> [(1 + x[1]^2)^2 + x[2]^2 - 4], [-8.75, -9.36]),

    # HS8 is a feasibility problem, so we limit to 3 instances.
    # BenchmarkEqualityProblem(2, x -> 1, x -> [], x -> [eq_sphere(x; radius = 5.0), x[1] * x[2] - 9], [2.0, 1.0]),
    # BenchmarkEqualityProblem(2, x -> 1, x -> [], x -> [eq_sphere(x; radius = 5.0), x[1] * x[2] - 9], [-6.0, 6.56]),
    # BenchmarkEqualityProblem(2, x -> 1, x -> [], x -> [eq_sphere(x; radius = 5.0), x[1] * x[2] - 9], [3.88, 3.18]),

    # HS26
    BenchmarkEqualityProblem(3, obj_hs26, x -> [], x -> [(1 + x[2]) * x[1] + x[3]^4 - 3], [-2.6, 2.0, 2.0]),
    BenchmarkEqualityProblem(3, obj_hs26, x -> [], x -> [(1 + x[2]) * x[1] + x[3]^4 - 3], [8.88, -23.66, 45.15]),
    BenchmarkEqualityProblem(3, obj_hs26, x -> [], x -> [(1 + x[2]) * x[1] + x[3]^4 - 3], [29.35, 2.99, 26.99]),
    BenchmarkEqualityProblem(3, obj_hs26, x -> [], x -> [(1 + x[2]) * x[1] + x[3]^4 - 3], [8.88, -23.66, 45.15]),
    BenchmarkEqualityProblem(3, obj_hs26, x -> [], x -> [(1 + x[2]) * x[1] + x[3]^4 - 3], [-22.05, 18.27, -28.64]),
    BenchmarkEqualityProblem(3, obj_hs26, x -> [], x -> [(1 + x[2]) * x[1] + x[3]^4 - 3], [10.69, 28.27, 16.57]),
    BenchmarkEqualityProblem(3, obj_hs26, x -> [], x -> [(1 + x[2]) * x[1] + x[3]^4 - 3], [-34.93, -44.83, -6.45]),
    BenchmarkEqualityProblem(3, obj_hs26, x -> [], x -> [(1 + x[2]) * x[1] + x[3]^4 - 3], [-4.45, 35.22, -43.95]),
    BenchmarkEqualityProblem(3, obj_hs26, x -> [], x -> [(1 + x[2]) * x[1] + x[3]^4 - 3], [4.71, 7.18, 29.28]),
    BenchmarkEqualityProblem(3, obj_hs26, x -> [], x -> [(1 + x[2]) * x[1] + x[3]^4 - 3], [11.26, 37.59, -46.18]),

    # HS27
    BenchmarkEqualityProblem(3, obj_hs27, x -> [], x -> [x[1] + x[3]^2 + 1], [2.0, 2.0, 2.0]),
    BenchmarkEqualityProblem(3, obj_hs27, x -> [], x -> [x[1] + x[3]^2 + 1], [11.88, 8.21, -11.87]),
    BenchmarkEqualityProblem(3, obj_hs27, x -> [], x -> [x[1] + x[3]^2 + 1], [-16.33, 18.54, 4.9]),
    BenchmarkEqualityProblem(3, obj_hs27, x -> [], x -> [x[1] + x[3]^2 + 1], [-12.86, -7.39, -3.28]),
    BenchmarkEqualityProblem(3, obj_hs27, x -> [], x -> [x[1] + x[3]^2 + 1], [18.54, 15.88, 11.03]),
    BenchmarkEqualityProblem(3, obj_hs27, x -> [], x -> [x[1] + x[3]^2 + 1], [5.15, -10.81, 12.33]),
    BenchmarkEqualityProblem(3, obj_hs27, x -> [], x -> [x[1] + x[3]^2 + 1], [-4.06, -16.49, -15.08]),

    # HS39
    BenchmarkEqualityProblem(4, x -> -x[1], x -> [], x -> [x[2] - x[1]^3 - x[3]^2, x[1]^2 - x[2] - x[4]^2], [2.0, 2.0, 2.0, 2.0, 2.0]),
    BenchmarkEqualityProblem(4, x -> -x[1], x -> [], x -> [x[2] - x[1]^3 - x[3]^2, x[1]^2 - x[2] - x[4]^2], [-11.63, -11.35, -14.47, 5.08]),
    BenchmarkEqualityProblem(4, x -> -x[1], x -> [], x -> [x[2] - x[1]^3 - x[3]^2, x[1]^2 - x[2] - x[4]^2], [14.72, -2.47, 2.54, -22.89]),
    BenchmarkEqualityProblem(4, x -> -x[1], x -> [], x -> [x[2] - x[1]^3 - x[3]^2, x[1]^2 - x[2] - x[4]^2], [21.23, 16.64, 9.53, 20.23]),
    BenchmarkEqualityProblem(4, x -> -x[1], x -> [], x -> [x[2] - x[1]^3 - x[3]^2, x[1]^2 - x[2] - x[4]^2], [-21.86, -18.99, 16.06, -13.21]),
    BenchmarkEqualityProblem(4, x -> -x[1], x -> [], x -> [x[2] - x[1]^3 - x[3]^2, x[1]^2 - x[2] - x[4]^2], [-5.11, 2.19, -4.21, 25.76]),
    BenchmarkEqualityProblem(4, x -> -x[1], x -> [], x -> [x[2] - x[1]^3 - x[3]^2, x[1]^2 - x[2] - x[4]^2], [0.15, -24.54, 21.43, -10.82]),

    # HS40
    BenchmarkEqualityProblem(4, x -> -prod(x), x -> [], x -> [x[1]^3 + x[2]^2 - 1, x[1]^2 * x[4] - x[3], x[4]^2 - x[2]], [0.8, 0.8, 0.8, 0.8]),
    BenchmarkEqualityProblem(4, x -> -prod(x), x -> [], x -> [x[1]^3 + x[2]^2 - 1, x[1]^2 * x[4] - x[3], x[4]^2 - x[2]], [5.89, -28.36, -19.94, -24.73]),
    BenchmarkEqualityProblem(4, x -> -prod(x), x -> [], x -> [x[1]^3 + x[2]^2 - 1, x[1]^2 * x[4] - x[3], x[4]^2 - x[2]], [23.21, 22.82, -14.08, 8.22]),
    BenchmarkEqualityProblem(4, x -> -prod(x), x -> [], x -> [x[1]^3 + x[2]^2 - 1, x[1]^2 * x[4] - x[3], x[4]^2 - x[2]], [-20.89, -14.41, 18.14, -14.19]),
    BenchmarkEqualityProblem(4, x -> -prod(x), x -> [], x -> [x[1]^3 + x[2]^2 - 1, x[1]^2 * x[4] - x[3], x[4]^2 - x[2]], [-7.3, 1.71, 10.26, -11.4]),
    BenchmarkEqualityProblem(4, x -> -prod(x), x -> [], x -> [x[1]^3 + x[2]^2 - 1, x[1]^2 * x[4] - x[3], x[4]^2 - x[2]], [-3.86, 13.39, 18.97, -0.45]),
    BenchmarkEqualityProblem(4, x -> -prod(x), x -> [], x -> [x[1]^3 + x[2]^2 - 1, x[1]^2 * x[4] - x[3], x[4]^2 - x[2]], [16.48, -10.14, -24.27, 1.74]),

    # HS42
    BenchmarkEqualityProblem(4, x -> sum([x[i] - i for i in 1:4]), x -> [], x -> [x[1] - 2, x[3]^2 + x[4]^2 - 2], [1.0, 1.0, 1.0, 1.0]),
    BenchmarkEqualityProblem(4, x -> sum([x[i] - i for i in 1:4]), x -> [], x -> [x[1] - 2, x[3]^2 + x[4]^2 - 2], [-6.74, -19.23, 14.02, -25.99]),
    BenchmarkEqualityProblem(4, x -> sum([x[i] - i for i in 1:4]), x -> [], x -> [x[1] - 2, x[3]^2 + x[4]^2 - 2], [28.49, -17.93, 10.12, 23.93]),
    BenchmarkEqualityProblem(4, x -> sum([x[i] - i for i in 1:4]), x -> [], x -> [x[1] - 2, x[3]^2 + x[4]^2 - 2], [-13.66, 2.4, -21.03, -15.26]),
    BenchmarkEqualityProblem(4, x -> sum([x[i] - i for i in 1:4]), x -> [], x -> [x[1] - 2, x[3]^2 + x[4]^2 - 2], [2.58, -6.09, -4.48, 29.8]),
    BenchmarkEqualityProblem(4, x -> sum([x[i] - i for i in 1:4]), x -> [], x -> [x[1] - 2, x[3]^2 + x[4]^2 - 2], [-27.19, -2.48, -27.45, 0.19]),
    BenchmarkEqualityProblem(4, x -> sum([x[i] - i for i in 1:4]), x -> [], x -> [x[1] - 2, x[3]^2 + x[4]^2 - 2], [10.94, 12.89, 29.72, -1.58]),

    # HS47
    BenchmarkEqualityProblem(5, obj_hs47, x -> [], x -> [x[1] + x[2]^2 + x[3]^3 - 3, x[2] - x[3]^2 + x[4] - 1, x[1] * x[5] - 1], [2.0, sqrt(2), -1, 2 - sqrt(2), 0.5]),
    BenchmarkEqualityProblem(5, obj_hs47, x -> [], x -> [x[1] + x[2]^2 + x[3]^3 - 3, x[2] - x[3]^2 + x[4] - 1, x[1] * x[5] - 1], [-28.99, -28.67, -19.83, -16.98, -26.97]),
    BenchmarkEqualityProblem(5, obj_hs47, x -> [], x -> [x[1] + x[2]^2 + x[3]^3 - 3, x[2] - x[3]^2 + x[4] - 1, x[1] * x[5] - 1], [13.43, 14.87, -6.89, 14.82, 28.42]),
    BenchmarkEqualityProblem(5, obj_hs47, x -> [], x -> [x[1] + x[2]^2 + x[3]^3 - 3, x[2] - x[3]^2 + x[4] - 1, x[1] * x[5] - 1], [27.22, 24.34, 10.26, 19.08, -13.19]),
    BenchmarkEqualityProblem(5, obj_hs47, x -> [], x -> [x[1] + x[2]^2 + x[3]^3 - 3, x[2] - x[3]^2 + x[4] - 1, x[1] * x[5] - 1], [-5.12, -19.33, -17.65, -2.49, 20.05]),
    BenchmarkEqualityProblem(5, obj_hs47, x -> [], x -> [x[1] + x[2]^2 + x[3]^3 - 3, x[2] - x[3]^2 + x[4] - 1, x[1] * x[5] - 1], [-10.69, -6.89, 25.01, 29.29, 10.77]),
    BenchmarkEqualityProblem(5, obj_hs47, x -> [], x -> [x[1] + x[2]^2 + x[3]^3 - 3, x[2] - x[3]^2 + x[4] - 1, x[1] * x[5] - 1], [1.18, 8.99, 3.01, 10.41, -19.88]),
    BenchmarkEqualityProblem(5, obj_hs47, x -> [], x -> [x[1] + x[2]^2 + x[3]^3 - 3, x[2] - x[3]^2 + x[4] - 1, x[1] * x[5] - 1], [-17.06, 23.59, 15.5, -18.72, -8.4]),

    # HS56
    BenchmarkEqualityProblem(7, x -> -x[1] * x[2] * x[3], x -> [], x -> [x[1] - 4.2 * sin(x[4])^2, x[2] - 4.2 * sin(x[5])^2, x[3] - 4.2 * sin(x[6])^2, x[1] + 2x[2] + 2x[3] - 7.2 * sin(x[7])^2], [1.0, 1.0, 1.0, asin(sqrt(1 / 4.2)), asin(sqrt(1 / 4.2)), asin(sqrt(1 / 4.2)), asin(sqrt(5 / 7.2))]),
    BenchmarkEqualityProblem(7, x -> -x[1] * x[2] * x[3], x -> [], x -> [x[1] - 4.2 * sin(x[4])^2, x[2] - 4.2 * sin(x[5])^2, x[3] - 4.2 * sin(x[6])^2, x[1] + 2x[2] + 2x[3] - 7.2 * sin(x[7])^2], [5.32, 3.49, 2.05, -3.67, -2.64, 1.53, 6.82]),
    BenchmarkEqualityProblem(7, x -> -x[1] * x[2] * x[3], x -> [], x -> [x[1] - 4.2 * sin(x[4])^2, x[2] - 4.2 * sin(x[5])^2, x[3] - 4.2 * sin(x[6])^2, x[1] + 2x[2] + 2x[3] - 7.2 * sin(x[7])^2], [10.07, -1.06, -1.77, 9.98, 4.54, -0.6, 3.85]),
    BenchmarkEqualityProblem(7, x -> -x[1] * x[2] * x[3], x -> [], x -> [x[1] - 4.2 * sin(x[4])^2, x[2] - 4.2 * sin(x[5])^2, x[3] - 4.2 * sin(x[6])^2, x[1] + 2x[2] + 2x[3] - 7.2 * sin(x[7])^2], [-2.24, -7.69, -11.39, -8.05, 11.28, 9.15, 7.17]),
    BenchmarkEqualityProblem(7, x -> -x[1] * x[2] * x[3], x -> [], x -> [x[1] - 4.2 * sin(x[4])^2, x[2] - 4.2 * sin(x[5])^2, x[3] - 4.2 * sin(x[6])^2, x[1] + 2x[2] + 2x[3] - 7.2 * sin(x[7])^2], [0.48, 6.81, 13.77, -11.03, 8.19, -13.32, -13.61]),
    BenchmarkEqualityProblem(7, x -> -x[1] * x[2] * x[3], x -> [], x -> [x[1] - 4.2 * sin(x[4])^2, x[2] - 4.2 * sin(x[5])^2, x[3] - 4.2 * sin(x[6])^2, x[1] + 2x[2] + 2x[3] - 7.2 * sin(x[7])^2], [-12.28, -3.42, -3.66, 0.89, -3.29, -6.47, -3.18]),
    BenchmarkEqualityProblem(7, x -> -x[1] * x[2] * x[3], x -> [], x -> [x[1] - 4.2 * sin(x[4])^2, x[2] - 4.2 * sin(x[5])^2, x[3] - 4.2 * sin(x[6])^2, x[1] + 2x[2] + 2x[3] - 7.2 * sin(x[7])^2], [7.0, 14.42, 8.73, -13.35, -8.85, 8.39, 10.41]),

    # HS60
    BenchmarkEqualityProblem(3, obj_hs60, x -> [[-x[i] - 10 for i in 1:3] ; [x[i] - 10 for i in 1:3]], x -> [x[1] * (1 + x[2]^2) + x[3]^4 - 4 - 3 * sqrt(2)], [2.0, 2.0, 2.0]),
    BenchmarkEqualityProblem(3, obj_hs60, x -> [[-x[i] - 10 for i in 1:3] ; [x[i] - 10 for i in 1:3]], x -> [x[1] * (1 + x[2]^2) + x[3]^4 - 4 - 3 * sqrt(2)], [-18.72, 0.76, 6.52]),
    BenchmarkEqualityProblem(3, obj_hs60, x -> [[-x[i] - 10 for i in 1:3] ; [x[i] - 10 for i in 1:3]], x -> [x[1] * (1 + x[2]^2) + x[3]^4 - 4 - 3 * sqrt(2)], [17.5, 24.62, 25.9]),
    BenchmarkEqualityProblem(3, obj_hs60, x -> [[-x[i] - 10 for i in 1:3] ; [x[i] - 10 for i in 1:3]], x -> [x[1] * (1 + x[2]^2) + x[3]^4 - 4 - 3 * sqrt(2)], [7.41, -27.75, -9.06]),
    BenchmarkEqualityProblem(3, obj_hs60, x -> [[-x[i] - 10 for i in 1:3] ; [x[i] - 10 for i in 1:3]], x -> [x[1] * (1 + x[2]^2) + x[3]^4 - 4 - 3 * sqrt(2)], [4.12, 11.77, -28.31]),
    BenchmarkEqualityProblem(3, obj_hs60, x -> [[-x[i] - 10 for i in 1:3] ; [x[i] - 10 for i in 1:3]], x -> [x[1] * (1 + x[2]^2) + x[3]^4 - 4 - 3 * sqrt(2)], [-9.7, -14.03, 4.09]),
    BenchmarkEqualityProblem(3, obj_hs60, x -> [[-x[i] - 10 for i in 1:3] ; [x[i] - 10 for i in 1:3]], x -> [x[1] * (1 + x[2]^2) + x[3]^4 - 4 - 3 * sqrt(2)], [7.75, -10.07, 7.65]),

    # HS61
    BenchmarkEqualityProblem(3, obj_hs61, x -> [], x -> [3x[1] - 2x[2]^2 - 7, 4x[1] - x[3]^2 - 11], [0.0, 0.0, 0.0]),
    BenchmarkEqualityProblem(3, obj_hs61, x -> [], x -> [3x[1] - 2x[2]^2 - 7, 4x[1] - x[3]^2 - 11], [12.66, -14.69, 9.34]),
    BenchmarkEqualityProblem(3, obj_hs61, x -> [], x -> [3x[1] - 2x[2]^2 - 7, 4x[1] - x[3]^2 - 11], [-12.29, -4.76, 7.11]),
    BenchmarkEqualityProblem(3, obj_hs61, x -> [], x -> [3x[1] - 2x[2]^2 - 7, 4x[1] - x[3]^2 - 11], [-0.12, 9.34, -12.28]),
    BenchmarkEqualityProblem(3, obj_hs61, x -> [], x -> [3x[1] - 2x[2]^2 - 7, 4x[1] - x[3]^2 - 11], [-5.59, -9.24, -6.61]),
    BenchmarkEqualityProblem(3, obj_hs61, x -> [], x -> [3x[1] - 2x[2]^2 - 7, 4x[1] - x[3]^2 - 11], [6.83, 6.9, -3.21]),
    BenchmarkEqualityProblem(3, obj_hs61, x -> [], x -> [3x[1] - 2x[2]^2 - 7, 4x[1] - x[3]^2 - 11], [-10.08, 13.67, -10.91]),

    # HS63
    BenchmarkEqualityProblem(3, obj_hs63, x -> -x, x -> [8x[1] + 14x[2] + 7x[3] - 56, eq_sphere(x; radius = 5.0)], [2.0, 2.0, 2.0]),
    BenchmarkEqualityProblem(3, obj_hs63, x -> -x, x -> [8x[1] + 14x[2] + 7x[3] - 56, eq_sphere(x; radius = 5.0)], [-6.69, 3.52, -5.69]),
    BenchmarkEqualityProblem(3, obj_hs63, x -> -x, x -> [8x[1] + 14x[2] + 7x[3] - 56, eq_sphere(x; radius = 5.0)], [8.37, 13.68, 9.61]),
    BenchmarkEqualityProblem(3, obj_hs63, x -> -x, x -> [8x[1] + 14x[2] + 7x[3] - 56, eq_sphere(x; radius = 5.0)], [-9.18, 8.8, -0.1]),
    BenchmarkEqualityProblem(3, obj_hs63, x -> -x, x -> [8x[1] + 14x[2] + 7x[3] - 56, eq_sphere(x; radius = 5.0)], [9.46, -1.34, 0.01]),
    BenchmarkEqualityProblem(3, obj_hs63, x -> -x, x -> [8x[1] + 14x[2] + 7x[3] - 56, eq_sphere(x; radius = 5.0)], [5.32, 0.31, -14.64]),
    BenchmarkEqualityProblem(3, obj_hs63, x -> -x, x -> [8x[1] + 14x[2] + 7x[3] - 56, eq_sphere(x; radius = 5.0)], [-2.17, -8.86, 7.65]),

    # HS71
    BenchmarkEqualityProblem(4, obj_hs71, x -> [[25 - prod(x)] ; 1 .- x ; x .- 5], x -> [eq_sphere(x; radius = sqrt(40))], [1.0, 5.0, 5.0, 1.0]),
    BenchmarkEqualityProblem(4, obj_hs71, x -> [[25 - prod(x)] ; 1 .- x ; x .- 5], x -> [eq_sphere(x; radius = sqrt(40))], [-6.57, -5.78, 2.53, -1.26]),
    BenchmarkEqualityProblem(4, obj_hs71, x -> [[25 - prod(x)] ; 1 .- x ; x .- 5], x -> [eq_sphere(x; radius = sqrt(40))], [2.86, -3.29, -6.5, -1.78]),

    # HS74
    BenchmarkEqualityProblem(4, obj_hs74, x -> [x[4] - x[3] + 0.55, x[3] - x[4] + 0.55, -x[1], -x[2], x[1] - 1200.0, x[2] - 1200.0, -0.55 - x[3], x[3] - 0.55, -0.55 - x[4], x[4] - 0.55], x -> [1000 * sin(-x[3] - 0.25) + 1000 * sin(-x[4] - 0.25) + 894.8 - x[1], 1000 * sin(x[3] - 0.25) + 1000 * sin(x[3] - x[4] - 0.25) + 894.8 - x[2], 1000 * sin(x[4] - 0.25) + 1000 * sin(x[4] - x[3] - 0.25) + 1294.8], [0.0, 0.0, 0.0, 0.0]),
    BenchmarkEqualityProblem(4, obj_hs74, x -> [x[4] - x[3] + 0.55, x[3] - x[4] + 0.55, -x[1], -x[2], x[1] - 1200.0, x[2] - 1200.0, -0.55 - x[3], x[3] - 0.55, -0.55 - x[4], x[4] - 0.55], x -> [1000 * sin(-x[3] - 0.25) + 1000 * sin(-x[4] - 0.25) + 894.8 - x[1], 1000 * sin(x[3] - 0.25) + 1000 * sin(x[3] - x[4] - 0.25) + 894.8 - x[2], 1000 * sin(x[4] - 0.25) + 1000 * sin(x[4] - x[3] - 0.25) + 1294.8], [400.0, 600.0, 0.12, -0.07, -0.4]),

    # HS77
    BenchmarkEqualityProblem(5, obj_hs77, x -> [], x -> [x[1]^2 * x[4] + sin(x[4] - x[5]) - 2 * sqrt(2), x[2] + x[3]^4 * x[4]^2 - 8 - sqrt(2)], [2.0, 2.0, 2.0, 2.0, 2.0]),
    BenchmarkEqualityProblem(5, obj_hs77, x -> [], x -> [x[1]^2 * x[4] + sin(x[4] - x[5]) - 2 * sqrt(2), x[2] + x[3]^4 * x[4]^2 - 8 - sqrt(2)], [-1.24, -6.38, 5.59, 6.3, -6.31]),
    BenchmarkEqualityProblem(5, obj_hs77, x -> [], x -> [x[1]^2 * x[4] + sin(x[4] - x[5]) - 2 * sqrt(2), x[2] + x[3]^4 * x[4]^2 - 8 - sqrt(2)], [3.43, 6.58, -1.59, 2.36, 2.87]),
    BenchmarkEqualityProblem(5, obj_hs77, x -> [], x -> [x[1]^2 * x[4] + sin(x[4] - x[5]) - 2 * sqrt(2), x[2] + x[3]^4 * x[4]^2 - 8 - sqrt(2)], [-4.71, 4.48, 6.56, 1.23, 5.06]),
    BenchmarkEqualityProblem(5, obj_hs77, x -> [], x -> [x[1]^2 * x[4] + sin(x[4] - x[5]) - 2 * sqrt(2), x[2] + x[3]^4 * x[4]^2 - 8 - sqrt(2)], [4.6, 4.15, -5.21, -1.02, 1.85]),

    # HS80 without its inequalities
    BenchmarkEqualityProblem(5, x -> exp(prod(x)), x -> [], eq_hs80, [-2.0, 2.0, 2.0, -1.0, -1.0]),

    # True HS80
    BenchmarkEqualityProblem(5, x -> exp(prod(x)), x -> [[-2.3 - x[i] for i in 1:2] ; [x[i] - 2.3 for i in 1:2] ; [-3.2 - x[i] for i in 3:5] ; [x[i] - 3.2 for i in 3:5]], eq_hs80, [-2.0, 2.0, 2.0, -1.0, -1.0]),

    # HS81
    BenchmarkEqualityProblem(5, obj_hs81, x -> [[-2.3 - x[i] for i in 1:2] ; [x[i] - 2.3 for i in 1:2] ; [-3.2 - x[i] for i in 3:5] ; [x[i] - 3.2 for i in 3:5]], eq_hs81, [-2.0, 2.0, 2.0, -1.0, -1.0]),
    BenchmarkEqualityProblem(5, obj_hs81, x -> [[-2.3 - x[i] for i in 1:2] ; [x[i] - 2.3 for i in 1:2] ; [-3.2 - x[i] for i in 3:5] ; [x[i] - 3.2 for i in 3:5]], eq_hs81, [-1.18, -1.57, 0.78, -0.37, -2.43]),

    # HS111
    BenchmarkEqualityProblem(10, obj_hs111, x -> [[-100 - x[i] for i in 1:10] ; [x[i] - 100 for i in 1:10]], eq_hs111, -2.3 .* ones(10)),
    BenchmarkEqualityProblem(10, obj_hs111, x -> [[-100 - x[i] for i in 1:10] ; [x[i] - 100 for i in 1:10]], eq_hs111, 2.3 .* ones(10)),
]


id_instance = parse(Int, ARGS[1])
instance = nonlinear_benchmarks[id_instance]

x_path = ARGS[2]
x = readdlm(x_path)[1, :]

f = eval_obj(instance, x)
h = eval_eqs(instance, x)

if has_inequality_constraints(instance)
    g = eval_ineqs(instance, x)
    print("$(f) ")
    for gi in g
        print("$(gi) ")
    end
    for hi in h[1:(end - 1)]
        print("$(hi) ")
    end
    println(h[end])
else
    print("$(f) ")
    for hi in h[1:(end - 1)]
        print("$(hi) ")
    end
    println(h[end])
end

using Manifolds
using Logging
using Random
using LinearAlgebra

function random_symmetric_generator(n::Int)
    A = rand(-10:10, n, n)
    return 0.5 .* (A .+ A')
end

# Objective functions
obj_axis(x; axis::Int = 1) = -x[axis]
obj_lin(x) = sum([i * x[i] for i in 1:length(x)])
obj_rosenbrock(x) = sum([100(x[i + 1] - x[i]^2)^2 + (1 - x[i])^2 for i in 1:(length(x) - 1)])
obj_prod(x) = prod([i * x[i] for i in 1:length(x)])
obj_spower(x) = sum([(x[i] - x[i + 1])^i for i in 1:(length(x) - 1)])
obj_rayleigh(x, A::Matrix{Float64}) = dot(x, (A * x)) / (norm(x)^2)
obj_hs54(x) = begin
    length(x) ≠ 6 && (@warn "Calling HS54 objective with dimension $(length(x)) point. Point should have dimension 6."; return 0)
    h = ((x[1] - 1.0e4)^2 / (6.4e7) + (x[1] - 1.0e4) * (x[2] - 1) / (2.0e4) + (x[2] - 1)^2) / 0.96 + 0.96 * ((x[3] - 2.0e6)^2) / (0.96 * 4.9e13) + ((x[4] - 10)^2) / (2.5e3) + ((x[5] - 1.0e-3)^2) / (2.5e-3) + ((x[6] - 1.0e8)^2) / (2.5e17)
    f = -exp(-h / 2)
    return f
end

export random_symmetric_generator, obj_axis, obj_lin, obj_rosenbrock, obj_prod, obj_spower, obj_rayleigh, obj_hs54

# Equality constraints
eq_sphere(x; radius::Float64 = 1.0) = sum(x .^ 2) - radius^2
eq_sum1(x) = sum(x) - 1.0

export eq_sphere, eq_sum1

linear_benchmark_sphere(dim::Int, x0::Vector{Float64}; use_manifolds::Bool = false) = begin
    obj(x) = sum([i * x[i] for i in 1:dim])
    ineqs(x) = []
    return use_manifolds ?
        BenchmarkManifoldProblem(
            Manifolds.Sphere(dim - 1),
            obj,
            ineqs,
            x0
        ) : BenchmarkEqualityProblem(
            dim,
            obj,
            ineqs,
            x -> [sum(x .^ 2) - 1],
            x0
        )
end

rosenbrock(dim::Int, x) = sum([100(x[i + 1] - x[i]^2)^2 + (1 - x[i])^2 for i in 1:(dim - 1)])

rosenbrock_sphere(dim::Int, x0::Vector{Float64}; use_manifolds::Bool = false) = begin
    obj = x -> sum([100(x[i + 1] - x[i]^2)^2 + (1 - x[i])^2 for i in 1:(dim - 1)])
    ineqs = x -> []
    return use_manifolds ?
        BenchmarkManifoldProblem(
            Manifolds.Sphere(dim - 1),
            obj,
            ineqs,
            x0
        ) : BenchmarkEqualityProblem(
            dim,
            obj,
            ineqs,
            x -> [sum(x .^ 2) - 1],
            x0
        )
end

############################
# From this point, this is the definitive list of instances used for benchmark in the thesis.
############################

linear_benchmarks = [
    # Rosenbrock function with sum-to-one constraint
    BenchmarkEqualityProblem(2, obj_rosenbrock, x -> [], x -> [eq_sum1(x)], [0.0, 0.0]),
    BenchmarkEqualityProblem(2, obj_rosenbrock, x -> [], x -> [eq_sum1(x)], [1.0, 0.0]),
    BenchmarkEqualityProblem(2, obj_rosenbrock, x -> [-x[1], -x[2]], x -> [eq_sum1(x)], (sqrt(2) / 2) .* ones(2)),
    BenchmarkEqualityProblem(2, obj_rosenbrock, x -> [], x -> [eq_sum1(x)], [1.0, 3.0]),
    BenchmarkEqualityProblem(3, obj_rosenbrock, x -> [], x -> [eq_sum1(x)], [0.0, 0.0, 0.0]),
    BenchmarkEqualityProblem(3, obj_rosenbrock, x -> [], x -> [eq_sum1(x)], [1.0, 0.0, 0.0]),
    BenchmarkEqualityProblem(3, obj_rosenbrock, x -> [-x[1], -x[2], -x[3]], x -> [eq_sum1(x)], (sqrt(3) / 3) .* ones(3)),
]

manifold_benchmarks = [

]

nonlinear_benchmarks = [

]

export linear_benchmark_sphere, rosenbrock_sphere, rosenbrock

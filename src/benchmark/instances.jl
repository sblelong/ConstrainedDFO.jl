using Manifolds
using ManifoldsBase
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
obj_hs48(x) = begin
    length(x) ≠ 5 && (@warn "Calling HS48 objective with dimension $(length(x)) point. Point should have dimension 5."; return 0)
    return (x[1] - 1)^2 + (x[2] - x[3])^2 + (x[4] - x[5])^2
end
obj_hs49(x) = begin
    length(x) ≠ 5 && (@warn "Calling HS49 objective with dimension $(length(x)) point. Point should have dimension 5."; return 0)
    return (x[1] - x[2])^2 + (x[3] - 1)^2 + (x[4] - 1)^4 + (x[5] - 1)^6
end
obj_hs51(x) = begin
    length(x) ≠ 5 && (@warn "Calling HS51 objective with dimension $(length(x)) point. Point should have dimension 5."; return 0)
    return (x[1] - x[2])^2 + (x[2] + x[3] - 2)^2 + (x[4] - 1)^2 + (x[5] - 1)^2
end
obj_hs52(x) = begin
    length(x) ≠ 5 && (@warn "Calling HS52 objective with dimension $(length(x)) point. Point should have dimension 5."; return 0)
    return (4x[1] - x[2])^2 + (x[2] + x[3] - 2)^2 + (x[4] - 1)^2 + (x[5] - 1)^2
end
obj_hs53(x) = begin
    length(x) ≠ 5 && (@warn "Calling HS53 objective with dimension $(length(x)) point. Point should have dimension 5."; return 0)
    return (x[1] - x[2])^2 + (x[2] + x[3] - 2)^2 + (x[4] - 1)^2 + (x[5] - 1)^2
end
obj_hs112(x) = begin
    length(x) ≠ 10 && (@warn "Calling HS112 objective with dimension $(length(x)) point. Point should have dimension 10."; return 0)
    c = [-6.089, -17.164, -34.054, -5.914, -24.721, -14.986, -24.1, -10.708, -26.662, -22.179]
    sx = sum(x)
    if all(x .≥ 0)
        f = sum([x[j] * (c[j] + log(x[j] / sx)) for j in 1:10])
    else
        f = typemax(Float64)
    end
    return f
end

A_hs119 = [
    1 0 0 1 0 0 1 1 0 0 0 0 0 0 0 1;
    0 1 1 0 0 0 1 0 0 1 0 0 0 0 0 0;
    0 0 1 0 0 0 1 0 1 1 0 0 0 1 0 0;
    0 0 0 1 0 0 1 0 0 0 1 0 0 0 1 0;
    0 0 0 0 1 1 0 0 0 1 0 1 0 0 0 1;
    0 0 0 0 0 1 0 1 0 0 0 0 0 0 1 0;
    0 0 0 0 0 0 1 0 0 0 1 0 1 0 0 0;
    0 0 0 0 0 0 0 1 0 1 0 0 0 0 1 0;
    0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 1;
    0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0;
    0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0;
    0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0;
    0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0;
    0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0;
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0;
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
]
obj_hs119(x) = begin
    length(x) ≠ 16 && (@warn "Calling HS119 objective with dimension $(length(x)) point. Point should have dimension 16."; return 0)
    return sum([sum([A_hs119[i, j] * (x[i]^2 + x[i] + 1) * (x[j]^2 + x[j] + 1) for j in 1:16]) for i in 1:16])
end

u_hs25(i) = 25 + (-50 * log(0.01 * i))^(2 / 3)
obj_hs25(x) = begin
    f = 1.0e20
    try
        f = sum([(-0.01 * i + exp(-1 / x[1] * (u_hs25(i) - x[2])^(x[3])))^2 for i in 1:99])
    catch e end
    return f
end

obj_hs62(x) = begin
    f = 1.0e20
    try
        f = -32.174 * (255 * log((x[1] + x[2] + x[3] + 0.03) / (0.09x[1] + x[2] + x[3] + 0.03)) + 280 * log((x[2] + x[3] + 0.03) / (0.07x[2] + x[3] + 0.03)) + 290 * log((x[3] + 0.03) / (0.13x[3] + 0.03)))
    catch e end
    return f
end

obj_hs110(x) = begin
    f = 1.0e20
    try
        f = sum([log(x[i] - 2)^2 + log(10 - x[i])^2 - prod(x)^2 for i in 1:10])
    catch e end
    return f
end

obj_hs5(x) = sin(x[1] + x[2]) + (x[1] - x[2])^2 - 1.5x[1] + 2.5x[2] + 1

export random_symmetric_generator, obj_axis, obj_lin, obj_rosenbrock, obj_prod, obj_spower, obj_rayleigh, obj_hs54, obj_hs48, obj_hs49, obj_hs51, obj_hs52, obj_hs53, obj_hs112, obj_hs119, obj_hs25, obj_hs62, obj_hs110, obj_hs5

# Equality constraints
eq_sphere(x; radius::Float64 = 1.0) = sum(x .^ 2) - radius^2
eq_sum1(x) = sum(x) - 1.0

B_hs119 = transpose(
    [
        0.22 -1.46  1.29 -1.1  0.0  0.0  1.12  0.0;
        0.2  0.0 -0.89 -1.06  0.0 -1.72  0.0  0.45;
        0.19 -1.3  0.0  0.95  0.0 -0.33  0.0  0.26;
        0.25  1.82  0.0 -0.54 -1.43  0.0  0.31 -1.1;
        0.15 -1.15 -1.16  0.0  1.51  1.62  0.0  0.58;
        0.11  0.0 -0.96 -1.78  0.59  1.24  0.0  0.0;
        0.12  0.8  0.0 -0.41 -0.33  0.21  1.12 -1.03;
        0.13  0.0 -0.49  0.0 -0.43 -0.26  0.0  0.1;
        1.0  0.0  0.0  0.0  0.0  0.0 -0.36  0.0;
        0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0;
        0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0;
        0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0;
        0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0;
        0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0;
        0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0;
        0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0
    ]
)
c_hs119 = [2.5, 1.1, -3.1, -3.5, 1.3, 2.1, 2.3, -1.5]
eq_hs119(x) = [sum([B_hs119[i, j] * x[j] for j in 1:16]) - c_hs119[i] for i in 1:8]
M_hs119 = EqualityManifold(eq_hs119, 8, 16)
starting_hs119_1 = project(M_hs119, 10 .* ones(16))
starting_hs119_2 = project(M_hs119, -5 .* ones(16))

export eq_sphere, eq_sum1, eq_hs119

# Some first test problems
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
    BenchmarkEqualityProblem(2, obj_rosenbrock, x -> [x[1]], x -> [eq_sum1(x)], 1 / 2 .* ones(2)),
    BenchmarkEqualityProblem(2, obj_rosenbrock, x -> [], x -> [eq_sum1(x)], [1.0, 3.0]),
    BenchmarkEqualityProblem(3, obj_rosenbrock, x -> [], x -> [eq_sum1(x)], [0.0, 0.0, 0.0]),
    BenchmarkEqualityProblem(3, obj_rosenbrock, x -> [], x -> [eq_sum1(x)], [1.0, 0.0, 0.0]),
    BenchmarkEqualityProblem(3, obj_rosenbrock, x -> [x[1]], x -> [eq_sum1(x)], 1 / 3 .* ones(3)),
    BenchmarkEqualityProblem(4, obj_rosenbrock, x -> [], x -> [eq_sum1(x)], zeros(4)),
    BenchmarkEqualityProblem(4, obj_rosenbrock, x -> [x[1], x[3]], x -> [eq_sum1(x)], 1 / 4 .* ones(4)),
    BenchmarkEqualityProblem(7, obj_rosenbrock, x -> [], x -> [eq_sum1(x)], 1 / 7 .* ones(7)),
    BenchmarkEqualityProblem(10, obj_rosenbrock, x -> [], x -> [eq_sum1(x)], 1 / 10 .* ones(10)),
    BenchmarkEqualityProblem(12, obj_rosenbrock, x -> [], x -> [eq_sum1(x)], 1 / 12 .* ones(12)),
    BenchmarkEqualityProblem(15, obj_rosenbrock, x -> [], x -> [eq_sum1(x)], 1 / 15 .* ones(15)),
    BenchmarkEqualityProblem(20, obj_rosenbrock, x -> [], x -> [eq_sum1(x)], 1 / 20 .* ones(20)),

    # Variant of HS48 with only the linear equality constraint
    BenchmarkEqualityProblem(5, obj_hs48, x -> [], x -> [sum(x) - 5], [3, 5, -3, 2, -2]),
    BenchmarkEqualityProblem(5, obj_hs48, x -> [], x -> [sum(x) - 5], [5, 0, 0, 0, 0]),
    BenchmarkEqualityProblem(5, obj_hs48, x -> [], x -> [sum(x) - 5], [2, 2, 1, 0, 0]),
    BenchmarkEqualityProblem(5, obj_hs48, x -> [], x -> [sum(x) - 5], [-2, 5, 3, 2, -3]),
    BenchmarkEqualityProblem(5, obj_hs48, x -> [], x -> [sum(x) - 5], [5, 3, -3, 2, -2]),
    BenchmarkEqualityProblem(5, obj_hs48, x -> [], x -> [sum(x) - 5], [-3, 2, 5, -2, 3]),
    BenchmarkEqualityProblem(5, obj_hs48, x -> [], x -> [sum(x) - 5], [-10, 5, 5, 5, 0]),

    # True HS49 with various starting points
    BenchmarkEqualityProblem(5, obj_hs49, x -> [], x -> [x[1] + x[2] + x[3] + 4x[4] - 7, x[3] + 5x[5] - 6], [10, 7, 2, -3, 0.8]),
    BenchmarkEqualityProblem(5, obj_hs49, x -> [], x -> [x[1] + x[2] + x[3] + 4x[4] - 7, x[3] + 5x[5] - 6], [6, 0, 1, 0, 1]),
    BenchmarkEqualityProblem(5, obj_hs49, x -> [], x -> [x[1] + x[2] + x[3] + 4x[4] - 7, x[3] + 5x[5] - 6], [-3, 0, 6, 1, 0]),
    BenchmarkEqualityProblem(5, obj_hs49, x -> [], x -> [x[1] + x[2] + x[3] + 4x[4] - 7, x[3] + 5x[5] - 6], [0, 1, 6, 0, 0]),
    BenchmarkEqualityProblem(5, obj_hs49, x -> [], x -> [x[1] + x[2] + x[3] + 4x[4] - 7, x[3] + 5x[5] - 6], [0, 1, 1, 1, 1]),

    # HS51 with various starting points
    BenchmarkEqualityProblem(5, obj_hs51, x -> [], x -> [x[1] + 3x[2] - 4, x[3] + x[4] - 2x[5], x[2] - x[5]], [2.5, 0.5, 2, -1, 0.5]),
    BenchmarkEqualityProblem(5, obj_hs51, x -> [], x -> [x[1] + 3x[2] - 4, x[3] + x[4] - 2x[5], x[2] - x[5]], [4, 0, -1, 1, 0]),
    BenchmarkEqualityProblem(5, obj_hs51, x -> [], x -> [x[1] + 3x[2] - 4, x[3] + x[4] - 2x[5], x[2] - x[5]], [1, 1, 2, 0, 1]),
    BenchmarkEqualityProblem(5, obj_hs51, x -> [], x -> [x[1] + 3x[2] - 4, x[3] + x[4] - 2x[5], x[2] - x[5]], [2.5, 0.5, 1.5, -0.5, 0.5]),

    # HS52 with feasible starting points
    BenchmarkEqualityProblem(5, obj_hs52, x -> [], x -> [x[1] + 3x[2], x[3] + x[4] - 2x[5], x[2] - x[5]], [0, 0, 1, 1, 0]),
    BenchmarkEqualityProblem(5, obj_hs52, x -> [], x -> [x[1] + 3x[2], x[3] + x[4] - 2x[5], x[2] - x[5]], [3, -1, -2, 0, 1]),
    BenchmarkEqualityProblem(5, obj_hs52, x -> [], x -> [x[1] + 3x[2], x[3] + x[4] - 2x[5], x[2] - x[5]], [3, -1, -1, 1, 1]),
    BenchmarkEqualityProblem(5, obj_hs52, x -> [], x -> [x[1] + 3x[2], x[3] + x[4] - 2x[5], x[2] - x[5]], [-3, 1, 3, 1, -1]),

    # HS53. This one has bounds.
    BenchmarkEqualityProblem(5, obj_hs53, x -> [[x[i] - 10 for i in 1:5] ; [-x[i] - 10 for i in 1:5]], x -> [x[1] + 3x[2], x[3] + x[4] - 2x[5], x[2] - x[5]], [0, 0, 1, 1, 0]),
    BenchmarkEqualityProblem(5, obj_hs53, x -> [[x[i] - 10 for i in 1:5] ; [-x[i] - 10 for i in 1:5]], x -> [x[1] + 3x[2], x[3] + x[4] - 2x[5], x[2] - x[5]], [3, -1, -2, 0, 1]),
    BenchmarkEqualityProblem(5, obj_hs53, x -> [[x[i] - 10 for i in 1:5] ; [-x[i] - 10 for i in 1:5]], x -> [x[1] + 3x[2], x[3] + x[4] - 2x[5], x[2] - x[5]], [3, -1, -1, 1, 1]),
    BenchmarkEqualityProblem(5, obj_hs53, x -> [[x[i] - 10 for i in 1:5] ; [-x[i] - 10 for i in 1:5]], x -> [x[1] + 3x[2], x[3] + x[4] - 2x[5], x[2] - x[5]], [-3, 1, 3, 1, -1]),

    # HS112
    BenchmarkEqualityProblem(10, obj_hs112, x -> [1.0e-6 - x[i] for i in 1:10], x -> [x[1] + 2x[2] + 2x[3] + x[6] + x[10] - 2, x[4] + 2x[5] + x[6] + x[7] - 1, x[3] + x[7] + x[8] + 2x[9] + x[10]], [3, 0, -1, 1, 0, 0, 0, 0, 0, 1]),
    BenchmarkEqualityProblem(10, obj_hs112, x -> [1.0e-6 - x[i] for i in 1:10], x -> [x[1] + 2x[2] + 2x[3] + x[6] + x[10] - 2, x[4] + 2x[5] + x[6] + x[7] - 1, x[3] + x[7] + x[8] + 2x[9] + x[10]], [0, 1, 0, 1, 0, 0, 0, 0, 0, 0]),

    # HS119
    BenchmarkEqualityProblem(16, obj_hs119, x -> [[-x[i] for i in 1:16] ; [x[i] - 5 for i in 1:16]], eq_hs119, starting_hs119_1),
    BenchmarkEqualityProblem(16, obj_hs119, x -> [[-x[i] for i in 1:16] ; [x[i] - 5 for i in 1:16]], eq_hs119, starting_hs119_2),
]

# Random symmetric matrices for extreme eigenvalues problems

A2 = [
    -10.0 1.0;
    1.0 7.0
]

A3 = [
    3.0 -0.5 2.5;
    -0.5 -3.0 2.5;
    2.5 2.5 -5.0
]

A5 = [
    2.0  -3.0  -2.0  -6.0  -4.0;
    -3.0   3.0  -3.5   1.5  -3.5;
    -2.0  -3.5   5.0   6.0   7.5;
    -6.0   1.5   6.0   7.0   4.5;
    -4.0  -3.5   7.5   4.5   4.0
]

A7 = [
    7.0  -2.0   7.5  -5.5   0.5  -2.5  -4.0;
    -2.0  -6.0  -4.0  -7.5  -5.0   2.0  -0.5;
    7.5  -4.0   5.0   1.5  -1.5   2.5   2.5;
    -5.5  -7.5   1.5  -8.0  -4.0   8.0   3.5;
    0.5  -5.0  -1.5  -4.0  -6.0  -1.0   2.5;
    -2.5   2.0   2.5   8.0  -1.0  -2.0  -3.5;
    -4.0  -0.5   2.5   3.5   2.5  -3.5  -8.0
]

A10 = [
    -5.0   5.5   3.5   1.0  -2.0  -3.0  -5.0   2.5   2.5   0.0;
    5.5   5.0   1.0  -5.0   5.0  -2.0   5.5   1.0  -1.0  -8.0;
    3.5   1.0   7.0   3.5  -0.5   5.5  -2.5  -5.5  -1.0   1.5;
    1.0  -5.0   3.5   6.0   5.0   4.0  -1.0   2.0  -6.0  -3.0;
    -2.0   5.0  -0.5   5.0   3.0  -5.0  -5.0   3.5   5.0  -7.0;
    -3.0  -2.0   5.5   4.0  -5.0  -8.0  -6.0   5.5  -4.0   1.5;
    -5.0   5.5  -2.5  -1.0  -5.0  -6.0  -3.0  -1.0   2.0   8.5;
    2.5   1.0  -5.5   2.0   3.5   5.5  -1.0   1.0   3.0   1.5;
    2.5  -1.0  -1.0  -6.0   5.0  -4.0   2.0   3.0   2.0   1.5;
    0.0  -8.0   1.5  -3.0  -7.0   1.5   8.5   1.5   1.5  -1.0
]

A15 = [
    -9.0  -0.5  -3.0  -2.0  -8.5    4.5    0.5   3.5  -8.5    3.0  -2.5   -5.0  -8.0  -0.5   5.0;
    -0.5  -1.0  -2.5   5.5   0.5   -4.0   -2.0   7.5  -7.0   -3.5  -5.0   -4.0   2.0   1.5   0.5;
    -3.0  -2.5   4.0   7.5  -0.5   -2.0    0.0  -8.0   1.0   -5.5  -0.5   -6.5  -3.0   2.0   3.5;
    -2.0   5.5   7.5   3.0   0.5   -2.0   -2.0  -6.0  -5.0    1.5   5.0    0.5   2.5  -1.5   4.5;
    -8.5   0.5  -0.5   0.5   6.0    5.0   -0.5   3.0  -3.0   -2.5  -0.5    0.5   9.5   2.5   1.0;
    4.5  -4.0  -2.0  -2.0   5.0   -5.0    0.0  -3.5  -2.0  -10.0  -0.5    0.5  -1.5  -3.0   4.0;
    0.5  -2.0   0.0  -2.0  -0.5    0.0  -10.0  -2.0   5.5   -3.0  -8.5   -1.0   1.0  -0.5  -8.0;
    3.5   7.5  -8.0  -6.0   3.0   -3.5   -2.0  10.0  -0.5    2.5   6.0    2.5  -1.0   2.5   0.0;
    -8.5  -7.0   1.0  -5.0  -3.0   -2.0    5.5  -0.5  -5.0    6.0  -7.0    1.5   6.0   5.0  -3.5;
    3.0  -3.5  -5.5   1.5  -2.5  -10.0   -3.0   2.5   6.0   -1.0   5.5    5.5   5.5   0.0   3.5;
    -2.5  -5.0  -0.5   5.0  -0.5   -0.5   -8.5   6.0  -7.0    5.5   2.0   -1.0   1.0  -0.5   3.5;
    -5.0  -4.0  -6.5   0.5   0.5    0.5   -1.0   2.5   1.5    5.5  -1.0  -10.0   2.5  -0.5   4.5;
    -8.0   2.0  -3.0   2.5   9.5   -1.5    1.0  -1.0   6.0    5.5   1.0    2.5  -4.0  -3.5   5.0;
    -0.5   1.5   2.0  -1.5   2.5   -3.0   -0.5   2.5   5.0    0.0  -0.5   -0.5  -3.5  -3.0   9.0;
    5.0   0.5   3.5   4.5   1.0    4.0   -8.0   0.0  -3.5    3.5   3.5    4.5   5.0   9.0  -3.0
]

A30 = [
    -7.0    5.0    -4.0    -6.0    6.0    0.0    7.5    8.0    -6.0    -4.5    -3.5    -6.0    4.0    -5.0    -0.5    -1.0    7.0    1.0    -3.5    1.5    4.5    -3.0    -5.5    -6.5    -2.5    2.0    -5.5    2.0    7.5    0.0;
    5.0    3.0    5.0    1.5    1.0    8.0    -0.5    3.5    3.5    -3.5    -3.0    5.5    2.5    -4.5    -6.5    -3.5    -5.5    -4.5    9.5    2.5    -1.0    -5.0    -1.0    4.0    6.5    5.5    0.0    7.5    -2.0    -7.5;
    -4.0    5.0    -5.0    6.0    -7.0    -2.0    1.0    5.0    2.0    3.0    -8.5    4.0    1.0    -1.5    3.5    -5.0    5.0    8.0    -0.5    -4.5    7.5    1.0    1.5    4.0    4.0    2.5    -3.5    -2.0    0.5    2.0;
    -6.0    1.5    6.0    -10.0    -1.0    -4.0    2.5    -7.5    -7.5    -7.5    0.5    3.0    0.0    -1.5    3.0    -3.0    6.5    7.5    2.0    -8.0    5.0    1.5    2.5    1.0    0.5    -3.0    2.5    2.0    -4.0    -6.0;
    6.0    1.0    -7.0    -1.0    -5.0    -1.0    1.5    3.5    -1.5    -5.5    3.0    -5.0    -5.0    1.0    5.0    -2.0    -1.5    -2.5    2.0    3.5    -9.0    -1.0    -4.0    2.5    1.5    -4.5    -7.5    1.5    -3.5    7.0;
    0.0    8.0    -2.0    -4.0    -1.0    -6.0    -1.0    -5.0    0.0    -1.5    -8.5    -4.5    -5.5    0.5    -5.0    5.5    4.0    5.0    -2.0    -7.0    -4.5    1.0    0.5    -7.0    4.5    -3.0    4.5    0.0    3.5    -4.0;
    7.5    -0.5    1.0    2.5    1.5    -1.0    -7.0    0.5    6.5    3.0    4.0    -0.5    -4.0    -2.5    -2.5    -0.5    4.5    0.5    5.0    8.5    -5.5    -1.0    1.0    -7.0    -6.5    4.5    -8.0    -5.0    -4.0    2.5;
    8.0    3.5    5.0    -7.5    3.5    -5.0    0.5    -8.0    4.0    -3.5    -7.0    5.5    -5.0    5.0    6.0    -1.5    -1.0    3.0    -0.5    1.5    -5.0    -4.5    -4.5    3.0    0.0    -7.0    -1.0    -6.5    1.5    -1.0;
    -6.0    3.5    2.0    -7.5    -1.5    0.0    6.5    4.0    8.0    7.5    0.0    3.0    -2.5    4.5    9.5    2.5    -0.5    -1.0    8.0    0.0    4.5    8.5    0.0    -0.5    -1.0    6.5    5.5    8.0    3.0    1.0;
    -4.5    -3.5    3.0    -7.5    -5.5    -1.5    3.0    -3.5    7.5    6.0    1.5    -2.5    0.5    1.0    0.5    8.5    -1.5    0.5    5.0    -5.5    8.0    1.5    1.5    -3.5    -5.0    -4.0    7.5    8.0    -5.0    9.0;
    -3.5    -3.0    -8.5    0.5    3.0    -8.5    4.0    -7.0    0.0    1.5    10.0    9.5    -1.0    -2.5    -4.0    1.5    6.5    -4.0    6.5    2.5    -0.5    0.5    -2.5    -8.5    -5.0    -6.0    -2.0    7.0    -2.0    -3.0;
    -6.0    5.5    4.0    3.0    -5.0    -4.5    -0.5    5.5    3.0    -2.5    9.5    1.0    4.5    -4.5    1.5    -4.5    -0.5    2.0    -4.0    -1.5    -4.5    -3.5    -4.0    1.5    4.0    -2.0    -3.5    -2.0    4.0    -1.0;
    4.0    2.5    1.0    0.0    -5.0    -5.5    -4.0    -5.0    -2.5    0.5    -1.0    4.5    -10.0    6.5    -4.0    3.0    0.0    2.0    -1.0    1.0    -3.0    -1.5    -1.5    -9.5    0.0    0.5    2.5    -3.0    -2.0    1.5;
    -5.0    -4.5    -1.5    -1.5    1.0    0.5    -2.5    5.0    4.5    1.0    -2.5    -4.5    6.5    -6.0    -6.0    5.0    6.0    1.0    -7.5    0.0    -0.5    -2.5    -0.5    2.0    0.5    9.5    0.0    7.0    -2.0    4.0;
    -0.5    -6.5    3.5    3.0    5.0    -5.0    -2.5    6.0    9.5    0.5    -4.0    1.5    -4.0    -6.0    8.0    7.0    -3.5    -9.5    2.5    -9.0    -4.0    2.5    0.0    -7.5    2.0    -5.0    1.0    0.5    4.5    -5.0;
    -1.0    -3.5    -5.0    -3.0    -2.0    5.5    -0.5    -1.5    2.5    8.5    1.5    -4.5    3.0    5.0    7.0    -3.0    0.0    -2.5    -7.0    2.0    -2.0    -6.0    -2.0    -2.0    4.5    -2.5    -4.5    -4.5    4.5    5.0;
    7.0    -5.5    5.0    6.5    -1.5    4.0    4.5    -1.0    -0.5    -1.5    6.5    -0.5    0.0    6.0    -3.5    0.0    -3.0    -1.0    0.5    -3.5    8.0    -4.0    -8.5    -2.0    -2.5    7.0    1.0    -1.5    -5.5    2.5;
    1.0    -4.5    8.0    7.5    -2.5    5.0    0.5    3.0    -1.0    0.5    -4.0    2.0    2.0    1.0    -9.5    -2.5    -1.0    -10.0    0.5    -1.5    -7.0    0.5    3.0    1.5    6.0    10.0    -0.5    -6.0    4.5    5.5;
    -3.5    9.5    -0.5    2.0    2.0    -2.0    5.0    -0.5    8.0    5.0    6.5    -4.0    -1.0    -7.5    2.5    -7.0    0.5    0.5    0.0    0.5    2.5    2.0    4.5    1.0    3.5    2.0    -2.5    7.0    -1.5    -1.5;
    1.5    2.5    -4.5    -8.0    3.5    -7.0    8.5    1.5    0.0    -5.5    2.5    -1.5    1.0    0.0    -9.0    2.0    -3.5    -1.5    0.5    -4.0    -4.0    0.0    1.0    -4.0    -3.0    -7.5    5.0    1.0    5.0    5.5;
    4.5    -1.0    7.5    5.0    -9.0    -4.5    -5.5    -5.0    4.5    8.0    -0.5    -4.5    -3.0    -0.5    -4.0    -2.0    8.0    -7.0    2.5    -4.0    3.0    -5.0    -1.5    9.0    -1.5    2.0    -2.0    -7.0    5.0    -2.5;
    -3.0    -5.0    1.0    1.5    -1.0    1.0    -1.0    -4.5    8.5    1.5    0.5    -3.5    -1.5    -2.5    2.5    -6.0    -4.0    0.5    2.0    0.0    -5.0    0.0    3.5    0.0    1.5    -6.5    -1.5    1.0    -3.5    9.0;
    -5.5    -1.0    1.5    2.5    -4.0    0.5    1.0    -4.5    0.0    1.5    -2.5    -4.0    -1.5    -0.5    0.0    -2.0    -8.5    3.0    4.5    1.0    -1.5    3.5    0.0    7.5    -7.0    5.5    3.0    -5.0    1.0    3.0;
    -6.5    4.0    4.0    1.0    2.5    -7.0    -7.0    3.0    -0.5    -3.5    -8.5    1.5    -9.5    2.0    -7.5    -2.0    -2.0    1.5    1.0    -4.0    9.0    0.0    7.5    -2.0    -6.0    0.0    4.5    -2.0    1.5    -2.5;
    -2.5    6.5    4.0    0.5    1.5    4.5    -6.5    0.0    -1.0    -5.0    -5.0    4.0    0.0    0.5    2.0    4.5    -2.5    6.0    3.5    -3.0    -1.5    1.5    -7.0    -6.0    6.0    -1.5    8.5    -3.0    2.0    5.5;
    2.0    5.5    2.5    -3.0    -4.5    -3.0    4.5    -7.0    6.5    -4.0    -6.0    -2.0    0.5    9.5    -5.0    -2.5    7.0    10.0    2.0    -7.5    2.0    -6.5    5.5    0.0    -1.5    5.0    -5.0    3.0    -7.0    5.0;
    -5.5    0.0    -3.5    2.5    -7.5    4.5    -8.0    -1.0    5.5    7.5    -2.0    -3.5    2.5    0.0    1.0    -4.5    1.0    -0.5    -2.5    5.0    -2.0    -1.5    3.0    4.5    8.5    -5.0    -3.0    8.0    1.5    -0.5;
    2.0    7.5    -2.0    2.0    1.5    0.0    -5.0    -6.5    8.0    8.0    7.0    -2.0    -3.0    7.0    0.5    -4.5    -1.5    -6.0    7.0    1.0    -7.0    1.0    -5.0    -2.0    -3.0    3.0    8.0    8.0    8.0    1.5;
    7.5    -2.0    0.5    -4.0    -3.5    3.5    -4.0    1.5    3.0    -5.0    -2.0    4.0    -2.0    -2.0    4.5    4.5    -5.5    4.5    -1.5    5.0    5.0    -3.5    1.0    1.5    2.0    -7.0    1.5    8.0    -6.0    8.5;
    0.0    -7.5    2.0    -6.0    7.0    -4.0    2.5    -1.0    1.0    9.0    -3.0    -1.0    1.5    4.0    -5.0    5.0    2.5    5.5    -1.5    5.5    -2.5    9.0    3.0    -2.5    5.5    5.0    -0.5    1.5    8.5    -7.0
]

manifold_benchmarks = [
    # Extreme eigenvalues on the unit sphere
    BenchmarkEqualityProblem(2, x -> obj_rayleigh(x, A2), x -> [], x -> [eq_sphere(x)], [1.0, 0.0]),
    BenchmarkEqualityProblem(2, x -> obj_rayleigh(x, A2), x -> [], x -> [eq_sphere(x)], [-1.0, -1.0]),
    BenchmarkEqualityProblem(2, x -> -obj_rayleigh(x, A2), x -> [], x -> [eq_sphere(x)], [1.0, 0.0]),
    BenchmarkEqualityProblem(2, x -> -obj_rayleigh(x, A2), x -> [], x -> [eq_sphere(x)], -ones(2)),

    BenchmarkEqualityProblem(3, x -> obj_rayleigh(x, A3), x -> [], x -> [eq_sphere(x)], [1.0, 0.0, 0.0]),
    BenchmarkEqualityProblem(3, x -> obj_rayleigh(x, A3), x -> [], x -> [eq_sphere(x)], -ones(3)),
    BenchmarkEqualityProblem(3, x -> -obj_rayleigh(x, A3), x -> [], x -> [eq_sphere(x)], [1.0, 0.0, 0.0]),
    BenchmarkEqualityProblem(3, x -> -obj_rayleigh(x, A3), x -> [], x -> [eq_sphere(x)], -ones(3)),

    BenchmarkEqualityProblem(5, x -> obj_rayleigh(x, A5), x -> [], x -> [eq_sphere(x)], [1.0, 0.0, 0.0, 0.0, 0.0]),
    BenchmarkEqualityProblem(5, x -> obj_rayleigh(x, A5), x -> [], x -> [eq_sphere(x)], -ones(5)),
    BenchmarkEqualityProblem(5, x -> -obj_rayleigh(x, A5), x -> [], x -> [eq_sphere(x)], [1.0, 0.0, 0.0, 0.0, 0.0]),
    BenchmarkEqualityProblem(5, x -> -obj_rayleigh(x, A5), x -> [], x -> [eq_sphere(x)], -ones(5)),

    BenchmarkEqualityProblem(7, x -> obj_rayleigh(x, A7), x -> [], x -> [eq_sphere(x)], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    BenchmarkEqualityProblem(7, x -> obj_rayleigh(x, A7), x -> [], x -> [eq_sphere(x)], -ones(7)),
    BenchmarkEqualityProblem(7, x -> -obj_rayleigh(x, A7), x -> [], x -> [eq_sphere(x)], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    BenchmarkEqualityProblem(7, x -> -obj_rayleigh(x, A7), x -> [], x -> [eq_sphere(x)], -ones(7)),

    BenchmarkEqualityProblem(10, x -> obj_rayleigh(x, A10), x -> [], x -> [eq_sphere(x)], [[1.0] ; [0.0 for _ in 1:9]]),
    BenchmarkEqualityProblem(10, x -> obj_rayleigh(x, A10), x -> [], x -> [eq_sphere(x)], -ones(10)),
    BenchmarkEqualityProblem(10, x -> -obj_rayleigh(x, A10), x -> [], x -> [eq_sphere(x)], [[1.0] ; [0.0 for _ in 1:9]]),
    BenchmarkEqualityProblem(10, x -> -obj_rayleigh(x, A10), x -> [], x -> [eq_sphere(x)], -ones(10)),
    BenchmarkEqualityProblem(10, x -> obj_rayleigh(x, A10), x -> [], x -> [eq_sphere(x)], ones(10)),
    BenchmarkEqualityProblem(10, x -> -obj_rayleigh(x, A10), x -> [], x -> [eq_sphere(x)], ones(10)),

    BenchmarkEqualityProblem(15, x -> obj_rayleigh(x, A15), x -> [], x -> [eq_sphere(x)], [[1.0] ; [0.0 for _ in 1:14]]),
    BenchmarkEqualityProblem(15, x -> obj_rayleigh(x, A15), x -> [], x -> [eq_sphere(x)], -ones(15)),
    BenchmarkEqualityProblem(15, x -> -obj_rayleigh(x, A15), x -> [], x -> [eq_sphere(x)], [[1.0] ; [0.0 for _ in 1:14]]),
    BenchmarkEqualityProblem(15, x -> -obj_rayleigh(x, A15), x -> [], x -> [eq_sphere(x)], -ones(15)),

    BenchmarkEqualityProblem(30, x -> obj_rayleigh(x, A30), x -> [], x -> [eq_sphere(x)], [[1.0] ; [0.0 for _ in 1:29]]),
    BenchmarkEqualityProblem(30, x -> -obj_rayleigh(x, A30), x -> [], x -> [eq_sphere(x)], -ones(30)),

    # Rosenbrock on spheres with various radii (divide by the norm for Manopt.jl)
    BenchmarkEqualityProblem(2, obj_rosenbrock, x -> [], x -> [eq_sphere(x)], [0.0, -1.0]),
    BenchmarkEqualityProblem(2, obj_rosenbrock, x -> [], x -> [eq_sphere(x)], [-2.17, 1.77]),
    BenchmarkEqualityProblem(3, obj_rosenbrock, x -> [], x -> [eq_sphere(x; radius = 4.0)], [0.0, -4.0, 0.0]),
    BenchmarkEqualityProblem(3, obj_rosenbrock, x -> [], x -> [eq_sphere(x; radius = 4.0)], [0.25, 3.48, 3.48]),
    BenchmarkEqualityProblem(5, obj_rosenbrock, x -> [], x -> [eq_sphere(x; radius = 2.0)], ones(5)),
    BenchmarkEqualityProblem(5, obj_rosenbrock, x -> [], x -> [eq_sphere(x; radius = 2.0)], [-3.38, -3.08, -1.06, -4.59, -3.98]),
    BenchmarkEqualityProblem(7, obj_rosenbrock, x -> [], x -> [eq_sphere(x; radius = 4.0)], [0.0, -4.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    BenchmarkEqualityProblem(15, obj_rosenbrock, x -> [], x -> [eq_sphere(x)], ones(15)),

    # Some HS objectives that could be restricted to spheres or ellipses
    # HS5
    BenchmarkEqualityProblem(2, obj_hs5, x -> [], x -> [eq_sphere(x)], [1.0, 0.0]),
    BenchmarkEqualityProblem(2, obj_hs5, x -> [], x -> [eq_sphere(x)], [4.49, 3.08]),

    # HS25
    BenchmarkEqualityProblem(3, obj_hs25, x -> [], x -> [eq_sphere(x; radius = 100.0)], [100, 12.5, 3]),
    BenchmarkEqualityProblem(3, obj_hs25, x -> [], x -> [eq_sphere(x; radius = 100.0)], [100, 12.5, 3]),
    BenchmarkEqualityProblem(3, obj_hs25, x -> [], x -> [eq_sphere(x; radius = 50.0)], [4.89, -2.88, -0.66]),
    BenchmarkEqualityProblem(3, obj_hs25, x -> [], x -> [eq_sphere(x; radius = 50.0)], -[100, 12.5, 3]),

    # HS54
    BenchmarkEqualityProblem(6, obj_hs54, x -> [], x -> [eq_sphere(x; radius = 1.0e8)], [6.0e3, 1.5, 4.0e6, 2, 3.0e-3, 5.0e7]),
    BenchmarkEqualityProblem(6, obj_hs54, x -> [], x -> [eq_sphere(x; radius = 1.0e8)], [1.0e8, 0.0, 0.0, 0.0, 0.0, 0.0]),

    # HS62
    BenchmarkEqualityProblem(3, obj_hs62, x -> [], x -> [eq_sphere(x)], [0.7, 0.2, 0.1]),
    BenchmarkEqualityProblem(3, obj_hs62, x -> [], x -> [eq_sphere(x; radius = 0.75)], [0.7, 0.2, 0.1]),
    BenchmarkEqualityProblem(3, obj_hs62, x -> [], x -> [eq_sphere(x; radius = 0.75)], [1.0, 0.0, -1.0]),

    # HS110
    BenchmarkEqualityProblem(10, obj_hs110, x -> [], x -> [eq_sphere(x; radius = 30.0)], 9 .* ones(10)),
    BenchmarkEqualityProblem(10, obj_hs110, x -> [], x -> [eq_sphere(x; radius = 30.0)], [[30.0] ; [0.0 for _ in 1:9]]),
]

nonlinear_benchmarks = [

    # HS48
    BenchmarkEqualityProblem(5, obj_hs48, x -> [], x -> [sum(x) - 5, x[3] - 2(x[4] - x[5]) + 3], [3, 5, -3, 2, -2]),
]

all_benchmarks = [linear_benchmarks ; manifold_benchmarks ; nonlinear_benchmarks]

export linear_benchmark_sphere, rosenbrock_sphere, rosenbrock, linear_benchmarks, manifold_benchmarks, nonlinear_benchmarks

using Manopt
using ConstrainedDFO
using Manifolds
using ProgressBars
using JSON

global objective_values_storage::Vector{Float64}

function objective_wrapper(obj::Function, x)
    global objective_values_storage
    f = obj(x)
    push!(objective_values_storage, f)
    return f
end

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

manifold_benchmarks_manopt = [
    BenchmarkManifoldProblem(Manifolds.Sphere(1), x -> objective_wrapper(y -> obj_rayleigh(y, A2), x), x -> [], [1.0, 0.0]),
    BenchmarkManifoldProblem(Manifolds.Sphere(1), x -> objective_wrapper(y -> obj_rayleigh(x, A2), x), x -> [], ones(2)),
    BenchmarkManifoldProblem(Manifolds.Sphere(1), x -> objective_wrapper(y -> -obj_rayleigh(x, A2), x), x -> [], [1.0, 0.0]),
    BenchmarkManifoldProblem(Manifolds.Sphere(1), x -> objective_wrapper(y -> -obj_rayleigh(x, A2), x), x -> [], -ones(2)),

    BenchmarkManifoldProblem(Manifolds.Sphere(2), x -> objective_wrapper(y -> obj_rayleigh(y, A3), x), x -> [], [1.0, 0.0, 0.0]),
    BenchmarkManifoldProblem(Manifolds.Sphere(2), x -> objective_wrapper(y -> obj_rayleigh(y, A3), x), x -> [], -ones(3)),
    BenchmarkManifoldProblem(Manifolds.Sphere(2), x -> objective_wrapper(y -> -obj_rayleigh(y, A3), x), x -> [], [1.0, 0.0, 0.0]),
    BenchmarkManifoldProblem(Manifolds.Sphere(2), x -> objective_wrapper(y -> -obj_rayleigh(y, A3), x), x -> [], -ones(3)),

    BenchmarkManifoldProblem(Manifolds.Sphere(4), x -> objective_wrapper(y -> obj_rayleigh(y, A5), x), x -> [], [1.0, 0.0, 0.0, 0.0, 0.0]),
    BenchmarkManifoldProblem(Manifolds.Sphere(4), x -> objective_wrapper(y -> obj_rayleigh(y, A5), x), x -> [], -ones(5)),
    BenchmarkManifoldProblem(Manifolds.Sphere(4), x -> objective_wrapper(y -> -obj_rayleigh(y, A5), x), x -> [], [1.0, 0.0, 0.0, 0.0, 0.0]),
    BenchmarkManifoldProblem(Manifolds.Sphere(4), x -> objective_wrapper(y -> -obj_rayleigh(y, A5), x), x -> [], -ones(5)),

    BenchmarkManifoldProblem(Manifolds.Sphere(6), x -> objective_wrapper(y -> obj_rayleigh(y, A7), x), x -> [], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    BenchmarkManifoldProblem(Manifolds.Sphere(6), x -> objective_wrapper(y -> obj_rayleigh(y, A7), x), x -> [], -ones(7)),
    BenchmarkManifoldProblem(Manifolds.Sphere(6), x -> objective_wrapper(y -> -obj_rayleigh(y, A7), x), x -> [], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    BenchmarkManifoldProblem(Manifolds.Sphere(6), x -> objective_wrapper(y -> -obj_rayleigh(y, A7), x), x -> [], -ones(7)),

    BenchmarkManifoldProblem(Manifolds.Sphere(9), x -> objective_wrapper(y -> obj_rayleigh(y, A10), x), x -> [], [[1.0] ; [0.0 for _ in 1:9]]),
    BenchmarkManifoldProblem(Manifolds.Sphere(9), x -> objective_wrapper(y -> obj_rayleigh(y, A10), x), x -> [], -ones(10)),
    BenchmarkManifoldProblem(Manifolds.Sphere(9), x -> objective_wrapper(y -> obj_rayleigh(y, A10), x), x -> [], [[1.0] ; [0.0 for _ in 1:9]]),
    BenchmarkManifoldProblem(Manifolds.Sphere(9), x -> objective_wrapper(y -> obj_rayleigh(y, A10), x), x -> [], -ones(10)),
    BenchmarkManifoldProblem(Manifolds.Sphere(9), x -> objective_wrapper(y -> obj_rayleigh(y, A10), x), x -> [], ones(10)),
    BenchmarkManifoldProblem(Manifolds.Sphere(9), x -> objective_wrapper(y -> obj_rayleigh(y, A10), x), x -> [], ones(10)),

    BenchmarkManifoldProblem(Manifolds.Sphere(14), x -> objective_wrapper(y -> obj_rayleigh(y, A15), x), x -> [], [[1.0] ; [0.0 for _ in 1:14]]),
    BenchmarkManifoldProblem(Manifolds.Sphere(14), x -> objective_wrapper(y -> obj_rayleigh(y, A15), x), x -> [], -ones(15)),
    BenchmarkManifoldProblem(Manifolds.Sphere(14), x -> objective_wrapper(y -> -obj_rayleigh(y, A15), x), x -> [], [[1.0] ; [0.0 for _ in 1:14]]),
    BenchmarkManifoldProblem(Manifolds.Sphere(14), x -> objective_wrapper(y -> -obj_rayleigh(y, A15), x), x -> [], -ones(15)),

    BenchmarkManifoldProblem(Manifolds.Sphere(29), x -> objective_wrapper(y -> obj_rayleigh(y, A30), x), x -> [], [[1.0] ; [0.0 for _ in 1:29]]),
    BenchmarkManifoldProblem(Manifolds.Sphere(29), x -> objective_wrapper(y -> -obj_rayleigh(y, A30), x), x -> [], -ones(30)),

    BenchmarkManifoldProblem(Manifolds.Sphere(1), x -> objective_wrapper(obj_rosenbrock, x), x -> [], [0.0, -1.0]),
    BenchmarkManifoldProblem(Manifolds.Sphere(1), x -> objective_wrapper(obj_rosenbrock, x), x -> [], [-2.17, 1.77]),
    BenchmarkManifoldProblem(ScaledSphere(2, 4.0), x -> objective_wrapper(obj_rosenbrock, x), x -> [], [0.0, -4.0, 0.0]),
    BenchmarkManifoldProblem(ScaledSphere(2, 4.0), x -> objective_wrapper(obj_rosenbrock, x), x -> [], [0.25, 3.48, 3.48]),
    BenchmarkManifoldProblem(ScaledSphere(4, 2.0), x -> objective_wrapper(obj_rosenbrock, x), x -> [], ones(5)),
    BenchmarkManifoldProblem(ScaledSphere(4, 2.0), x -> objective_wrapper(obj_rosenbrock, x), x -> [], [-3.38, -3.08, -1.06, -4.59, -3.98]),
    BenchmarkManifoldProblem(ScaledSphere(6, 4.0), x -> objective_wrapper(obj_rosenbrock, x), x -> [], [0.0, -4.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    BenchmarkManifoldProblem(Manifolds.Sphere(14), x -> objective_wrapper(obj_rosenbrock, x), x -> [], ones(15)),

    BenchmarkManifoldProblem(Manifolds.Sphere(1), x -> objective_wrapper(obj_hs5, x), x -> [], [1.0, 0.0]),
    BenchmarkManifoldProblem(Manifolds.Sphere(1), x -> objective_wrapper(obj_hs5, x), x -> [], [4.49, 3.08]),

    BenchmarkManifoldProblem(ScaledSphere(2, 100.0), x -> objective_wrapper(obj_hs25, x), x -> [], [100.0, 12.5, 3.0]),
    BenchmarkManifoldProblem(ScaledSphere(2, 100.0), x -> objective_wrapper(obj_hs25, x), x -> [], [-78.2, 9.1, -15.6]),
    BenchmarkManifoldProblem(ScaledSphere(2, 50.0), x -> objective_wrapper(obj_hs25, x), x -> [], [4.89, -2.88, -0.66]),
    BenchmarkManifoldProblem(ScaledSphere(2, 50.0), x -> objective_wrapper(obj_hs25, x), x -> [], -[100.0, 12.5, 3.0]),

    BenchmarkManifoldProblem(ScaledSphere(5, 1.0e8), x -> objective_wrapper(obj_hs54, x), x -> [], [6.0e3, 1.5, 4.0e6, 2, 3.0e-3, 5.0e7]),
    BenchmarkManifoldProblem(ScaledSphere(5, 1.0e8), x -> objective_wrapper(obj_hs54, x), x -> [], [1.0e8, 0.0, 0.0, 0.0, 0.0, 0.0]),

    BenchmarkManifoldProblem(Manifolds.Sphere(2), x -> objective_wrapper(obj_hs62, x), x -> [], [0.7, 0.2, 0.1]),
    BenchmarkManifoldProblem(ScaledSphere(2, 0.75), x -> objective_wrapper(obj_hs62, x), x -> [], [0.7, 0.2, 0.1]),
    BenchmarkManifoldProblem(ScaledSphere(2, 0.75), x -> objective_wrapper(obj_hs62, x), x -> [], [1.0, 0.0, -1.0]),

    BenchmarkManifoldProblem(ScaledSphere(9, 30.0), x -> objective_wrapper(obj_hs110, x), x -> [], 9 .* ones(10)),
    BenchmarkManifoldProblem(ScaledSphere(9, 30.0), x -> objective_wrapper(obj_hs110, x), x -> [], [[30.0] ; [0.0 for _ in 1:9]]),
]

data_path = "/home/sblelong/.julia/dev/ConstrainedDFO/src/benchmark/4-manifolds/data/manopt"

for (i, problem) in ProgressBar(enumerate(manifold_benchmarks_manopt))
    global objective_values_storage
    objective_values_storage = Float64[]

    M = get_equality_manifold(problem)
    obj(M, x) = eval_obj(problem, x)
    x0 = get_x0(problem)

    result = mesh_adaptive_direct_search(M, obj, x0; stopping_criterion = StopAfterIteration(1000 * get_dimension(problem)) | StopWhenPollSizeLess(1.0e-10))

    stratified_f = Float64[]
    best = typemax(Float64)
    for f_val in objective_values_storage
        best = min(best, f_val)
        push!(stratified_f, best)
    end

    data = Dict(
        "stratified_f" => stratified_f
    )
    open(joinpath(data_path, "$(i).json"), "w") do io
        JSON.print(io, data)
    end
end

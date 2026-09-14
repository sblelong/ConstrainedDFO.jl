using ConstrainedDFO
using Manifolds
using ProgressBars
using JSON

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

manifold_benchmarks_manifolds = [
    BenchmarkManifoldProblem(Manifolds.Sphere(1), x -> obj_rayleigh(x, A2), x -> [], [1.0, 0.0]),
    BenchmarkManifoldProblem(Manifolds.Sphere(1), x -> obj_rayleigh(x, A2), x -> [], ones(2)),
    BenchmarkManifoldProblem(Manifolds.Sphere(1), x -> -obj_rayleigh(x, A2), x -> [], [1.0, 0.0]),
    BenchmarkManifoldProblem(Manifolds.Sphere(1), x -> -obj_rayleigh(x, A2), x -> [], -ones(2)),

    BenchmarkManifoldProblem(Manifolds.Sphere(2), x -> obj_rayleigh(x, A3), x -> [], [1.0, 0.0, 0.0]),
    BenchmarkManifoldProblem(Manifolds.Sphere(2), x -> obj_rayleigh(x, A3), x -> [], -ones(3)),
    BenchmarkManifoldProblem(Manifolds.Sphere(2), x -> -obj_rayleigh(x, A3), x -> [], [1.0, 0.0, 0.0]),
    BenchmarkManifoldProblem(Manifolds.Sphere(2), x -> -obj_rayleigh(x, A3), x -> [], -ones(3)),

    BenchmarkManifoldProblem(Manifolds.Sphere(4), x -> obj_rayleigh(x, A5), x -> [], [1.0, 0.0, 0.0, 0.0, 0.0]),
    BenchmarkManifoldProblem(Manifolds.Sphere(4), x -> obj_rayleigh(x, A5), x -> [], -ones(5)),
    BenchmarkManifoldProblem(Manifolds.Sphere(4), x -> -obj_rayleigh(x, A5), x -> [], [1.0, 0.0, 0.0, 0.0, 0.0]),
    BenchmarkManifoldProblem(Manifolds.Sphere(4), x -> -obj_rayleigh(x, A5), x -> [], -ones(5)),

    BenchmarkManifoldProblem(Manifolds.Sphere(6), x -> obj_rayleigh(x, A7), x -> [], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    BenchmarkManifoldProblem(Manifolds.Sphere(6), x -> obj_rayleigh(x, A7), x -> [], -ones(7)),
    BenchmarkManifoldProblem(Manifolds.Sphere(6), x -> -obj_rayleigh(x, A7), x -> [], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    BenchmarkManifoldProblem(Manifolds.Sphere(6), x -> -obj_rayleigh(x, A7), x -> [], -ones(7)),

    BenchmarkManifoldProblem(Manifolds.Sphere(9), x -> obj_rayleigh(x, A10), x -> [], [[1.0] ; [0.0 for _ in 1:9]]),
    BenchmarkManifoldProblem(Manifolds.Sphere(9), x -> obj_rayleigh(x, A10), x -> [], -ones(10)),
    BenchmarkManifoldProblem(Manifolds.Sphere(9), x -> -obj_rayleigh(x, A10), x -> [], [[1.0] ; [0.0 for _ in 1:9]]),
    BenchmarkManifoldProblem(Manifolds.Sphere(9), x -> -obj_rayleigh(x, A10), x -> [], -ones(10)),
    BenchmarkManifoldProblem(Manifolds.Sphere(9), x -> obj_rayleigh(x, A10), x -> [], ones(10)),
    BenchmarkManifoldProblem(Manifolds.Sphere(9), x -> -obj_rayleigh(x, A10), x -> [], ones(10)),

    BenchmarkManifoldProblem(Manifolds.Sphere(14), x -> obj_rayleigh(x, A15), x -> [], [[1.0] ; [0.0 for _ in 1:14]]),
    BenchmarkManifoldProblem(Manifolds.Sphere(14), x -> obj_rayleigh(x, A15), x -> [], -ones(15)),
    BenchmarkManifoldProblem(Manifolds.Sphere(14), x -> -obj_rayleigh(x, A15), x -> [], [[1.0] ; [0.0 for _ in 1:14]]),
    BenchmarkManifoldProblem(Manifolds.Sphere(14), x -> -obj_rayleigh(x, A15), x -> [], -ones(15)),

    BenchmarkManifoldProblem(Manifolds.Sphere(29), x -> obj_rayleigh(x, A30), x -> [], [[1.0] ; [0.0 for _ in 1:29]]),
    BenchmarkManifoldProblem(Manifolds.Sphere(29), x -> -obj_rayleigh(x, A30), x -> [], -ones(30)),

    BenchmarkManifoldProblem(Manifolds.Sphere(1), obj_rosenbrock, x -> [], [0.0, -1.0]),
    BenchmarkManifoldProblem(Manifolds.Sphere(1), obj_rosenbrock, x -> [], [-2.17, 1.77]),
    BenchmarkManifoldProblem(ScaledSphere(2, 4.0), obj_rosenbrock, x -> [], [0.0, -4.0, 0.0]),
    BenchmarkManifoldProblem(ScaledSphere(2, 4.0), obj_rosenbrock, x -> [], [0.25, 3.48, 3.48]),
    BenchmarkManifoldProblem(ScaledSphere(4, 2.0), obj_rosenbrock, x -> [], ones(5)),
    BenchmarkManifoldProblem(ScaledSphere(4, 2.0), obj_rosenbrock, x -> [], [-3.38, -3.08, -1.06, -4.59, -3.98]),
    BenchmarkManifoldProblem(ScaledSphere(6, 4.0), obj_rosenbrock, x -> [], [0.0, -4.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    BenchmarkManifoldProblem(Manifolds.Sphere(14), obj_rosenbrock, x -> [], ones(15)),

    BenchmarkManifoldProblem(Manifolds.Sphere(1), obj_hs5, x -> [], [1.0, 0.0]),
    BenchmarkManifoldProblem(Manifolds.Sphere(1), obj_hs5, x -> [], [4.49, 3.08]),

    BenchmarkManifoldProblem(ScaledSphere(2, 100.0), obj_hs25, x -> [], [100.0, 12.5, 3.0]),
    BenchmarkManifoldProblem(ScaledSphere(2, 100.0), obj_hs25, x -> [], [-78.2, 9.1, -15.6]),
    BenchmarkManifoldProblem(ScaledSphere(2, 50.0), obj_hs25, x -> [], [4.89, -2.88, -0.66]),
    BenchmarkManifoldProblem(ScaledSphere(2, 50.0), obj_hs25, x -> [], -[100.0, 12.5, 3.0]),

    BenchmarkManifoldProblem(ScaledSphere(5, 1.0e8), obj_hs54, x -> [], [6.0e3, 1.5, 4.0e6, 2, 3.0e-3, 5.0e7]),
    BenchmarkManifoldProblem(ScaledSphere(5, 1.0e8), obj_hs54, x -> [], [1.0e8, 0.0, 0.0, 0.0, 0.0, 0.0]),

    BenchmarkManifoldProblem(Manifolds.Sphere(2), obj_hs62, x -> [], [0.7, 0.2, 0.1]),
    BenchmarkManifoldProblem(ScaledSphere(2, 0.75), obj_hs62, x -> [], [0.7, 0.2, 0.1]),
    BenchmarkManifoldProblem(ScaledSphere(2, 0.75), obj_hs62, x -> [], [1.0, 0.0, -1.0]),

    BenchmarkManifoldProblem(ScaledSphere(9, 30.0), obj_hs110, x -> [], 9 .* ones(10)),
    BenchmarkManifoldProblem(ScaledSphere(9, 30.0), obj_hs110, x -> [], [[30.0] ; [0.0 for _ in 1:9]]),
]

data_path = "/home/sblelong/.julia/dev/ConstrainedDFO/src/benchmark/2-retractions/data/Exp/"

for (i, problem) in ProgressBar(enumerate(manifold_benchmarks_manifolds))
    result, stratified_f, x_history, f_history, v_history, d_history = solve_problem(RDFOSolver(), problem)
    data = Dict(
        "stratified_f" => stratified_f
    )
    open(joinpath(data_path, "$(i).json"), "w") do io
        JSON.print(io, data)
    end
end

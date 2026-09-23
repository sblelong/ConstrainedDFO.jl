using ConstrainedDFO
using Manifolds, Manopt
using LinearAlgebra
using Printf

"""
Add a unit sphere constraint to an instance and solve it with the MADS solver implemented in Manopt.
"""
function solve_sphere_manopt(BI::BlackboxInstance; max_evals::Int = 1000 * (get_dimension(BI) + 1), tol_eqs::Float64 = 1.0e-8)
    dimension = get_dimension(BI)
    M = Manifolds.Sphere(dimension - 1)

    x0 = ConstrainedDFO.get_x0(BI)
    p0 = norm(x0) == 0 ? [[1.0] ; [0.0 for _ in 1:(dimension - 1)]] : x0 ./ norm(x0)

    function f(M::Manifolds.Sphere, p)
        hval = norm(p)^2 - 1.0
        fval = hval ≤ tol_eqs ? eval_objective(BI, p) : ConstrainedDFO.FAILURE_MAX
        line = @sprintf("%-20.10g%-20.10g", fval, hval)
        println(line)
        return fval
    end

    stopping_criterion = StopAfterIteration(max_evals) | StopWhenPollSizeLess(1.0e-10)

    result_manopt = mesh_adaptive_direct_search(M, f, p0; stopping_criterion = stopping_criterion)
    return result_manopt
end

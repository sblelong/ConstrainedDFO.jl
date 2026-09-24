using ConstrainedDFO
using NOMAD

"""
Solve a problem with the NOMAD solver by handling equality constraints with a progressive or extreme barrier as |h(x)|<=0.
"""
function solve_nomad(BI::BlackboxInstance; barrier::Symbol = :PB, max_evals::Int = 1000 * (get_dimension(BI) + 1), tol_eqs::Float64 = 1.0e-8)
    dimension = get_dimension(BI)
    n_eqs = get_n_eqs(BI)
    n_ineqs = get_n_ineqs(BI)
    n_constraints = n_eqs + n_ineqs
    output_types = [["OBJ"] ; [String(barrier) for _ in 1:n_constraints]]
    x0 = ConstrainedDFO.get_x0(BI)

    # CAREFUL!! This is temporary: NOMAD knows how to handle bounds, when adding inequalities this should be taken into account.
    function blackbox(x)
        f = eval_objective(BI, x)
        h = eval_eqs(BI, x)
        g = eval_ineqs(BI, x)
        c = [abs.(h) .- tol_eqs ; g]
        return (true, true, [[f] ; c])
    end

    options = NOMAD.NomadOptions(max_bb_eval = max_evals, display_stats = ["BBE", "BBO"], display_all_eval = true)
    pb = NomadProblem(dimension, 1 + n_constraints, output_types, blackbox; options = options)
    return result = solve(pb, x0)
end

function solve_nomad_converter(BI::BlackboxInstance, A::Matrix{Float64}, b::Vector{Float64}; converter = :SVD, barrier = :PB, max_evals = 1000 * (get_dimension(BI) + 1), tol_eqs::Float64 = 1.0e-8)
    dimension = get_dimension(BI)
    n_ineqs = get_n_ineqs(BI)
    output_types = [["OBJ"] ; [String(barrier) for _ in 1:n_ineqs]]
    x0 = ConstrainedDFO.get_x0(BI)
    lb = get_lower_bounds(BI)
    ub = get_upper_bounds(BI)

    function blackbox(x)
        h = eval_eqs(BI, x)
        f = isapprox(h, 0.0; atol = tol_eqs) ? eval_objective(BI, x) : ConstrainedDFO.FAILURE_MAX
        g = eval_ineqs(BI, x)
        return (true, true, [[f] ; g])
    end

    options = NOMAD.NomadOptions(max_bb_eval = max_evals, display_stats = ["BBE", "BBO"], display_all_eval = true, linear_converter = String(converter), linear_constraints_atol = tol_eqs)
    pb = NomadProblem(dimension, 1 + n_ineqs, output_types, blackbox; A = A, b = b, lower_bound = lb, upper_bound = ub, options = options)
    return solve(pb, x0)
end

using ConstrainedDFO
using NOMAD

"""
Solve a problem with the NOMAD solver by considering equality constraints as double inequalities.
"""
function solve_nomad(BI::BlackboxInstance; barrier::Symbol = :PB, max_evals::Int = 1000 * (get_dimension(BI) + 1), tol_eqs::Float64 = 1.0e-8)
    dimension = get_dimension(BI)
    n_eqs = get_n_eqs(BI)
    n_ineqs = get_n_ineqs(BI)
    n_constraints = n_eqs + n_ineqs # Equalities = double inequalities + inequalities
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

    options = NOMAD.NomadOptions(max_bb_eval = max_evals, display_stats = ["BBE", "SOL", "BBO"], display_all_eval = true)
    pb = NomadProblem(dimension, 1 + n_constraints, output_types, blackbox; options = options)
    return result = solve(pb, x0)
end

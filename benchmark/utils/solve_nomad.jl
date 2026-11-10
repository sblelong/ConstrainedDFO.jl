using ConstrainedDFO
using NOMAD
using NLPModels
using Printf

include(joinpath(@__DIR__, "project.jl"))

"""
Prints an evaluation as
BBE OBJ CONS
with 15 significant digits, since NOMAD's own display is limited to 6 decimals.
NOMAD does not call the blackbox on cache hits, which are not counted as evaluations either.
The line is written directly to the stdout file descriptor, as NOMAD does: the blackbox is a C callback, from which Julia's IO is not safe.
"""
function print_nomad_evaluation(bbe::Int, values::Vector{Float64})
    line = "$(bbe) " * join((@sprintf("%.15g", v) for v in values), " ") * "\n"
    ccall(:write, Cssize_t, (Cint, Ptr{UInt8}, Csize_t), 1, line, sizeof(line))
    return
end

"""
Solve a problem with the NOMAD solver by handling equality constraints with a progressive or extreme barrier as |h(x)|<=0.
"""
function solve_nomad(BI::BlackboxInstance; x0::Vector{Float64} = ConstrainedDFO.get_x0(BI), barrier::Symbol = :PB, max_evals::Int = 200 * (get_dimension(BI) + 1), tol_eqs::Float64 = 1.0e-8)
    dimension = get_dimension(BI)
    n_eqs = get_n_eqs(BI)
    n_ineqs = get_n_ineqs(BI)
    n_constraints = n_eqs + n_ineqs
    output_types = [["OBJ"] ; [String(barrier) for _ in 1:n_constraints]]
    lbounds, ubounds = get_lbounds(BI), get_ubounds(BI)

    n_evals = 0
    function blackbox(x)
        f = eval_objective(BI, x)
        h = eval_eqs(BI, x)
        g = eval_ineqs(BI, x)
        n_evals += 1
        print_nomad_evaluation(n_evals, [[f] ; h ; g]) # The equality constraints are logged as is, not as |h(x)|-tol_eqs
        c = [abs.(h) .- tol_eqs ; g]
        return (true, true, [[f] ; c])
    end

    options = NOMAD.NomadOptions(max_bb_eval = max_evals, display_degree = 0)
    pb = NomadProblem(dimension, 1 + n_constraints, output_types, blackbox; lower_bound = lbounds, upper_bound = ubounds, options = options)
    return solve(pb, x0)
end

function solve_nomad_converter(BI::BlackboxInstance, A::Matrix{Float64}, b::Vector{Float64}; x0::Vector{Float64} = ConstrainedDFO.get_x0(BI), converter = :SVD, barrier = :PB, max_evals = 1000 * (get_dimension(BI) + 1), tol_eqs::Float64 = 1.0e-8)
    dimension = get_dimension(BI)
    n_ineqs = get_n_ineqs(BI)
    output_types = [["OBJ"] ; [String(barrier) for _ in 1:n_ineqs]]
    lb = get_lbounds(BI)
    ub = get_ubounds(BI)

    n_evals = 0
    function blackbox(x)
        h = eval_eqs(BI, x)
        if all(isapprox.(h, 0.0; atol = tol_eqs))
            f = eval_objective(BI, x)
            g = eval_ineqs(BI, x)
        else
            f = FAILURE_MAX
            g = fill(FAILURE_MAX, n_ineqs)
        end
        n_evals += 1
        print_nomad_evaluation(n_evals, [[f] ; g])
        return (true, true, [[f] ; g])
    end

    reduced_dimension = dimension - size(A)[1]
    eval_queue_sort = reduced_dimension ≥ 50 ? "DIR_LAST_SUCCESS" : "QUADRATIC_MODEL"
    # NOMAD.jl checks norm(A*x0 - b) <= atol, whereas the blackbox checks each |h_i(x)| <= tol_eqs: scale by sqrt(m) to accept every point feasible for the blackbox
    options = NOMAD.NomadOptions(max_bb_eval = max_evals, display_degree = 0, linear_converter = String(converter), linear_constraints_atol = sqrt(size(A, 1)) * tol_eqs, eval_queue_sort = eval_queue_sort)
    pb = NomadProblem(dimension, 1 + n_ineqs, output_types, blackbox; A = A, b = b, lower_bound = lb, upper_bound = ub, options = options)
    return solve(pb, x0)
end

function solve_mads_pip(BI::BlackboxInstance; x0::Vector{Float64} = ConstrainedDFO.get_x0(BI), max_evals = 200 * (get_dimension(BI) + 1))
    dimension = get_dimension(BI)
    lbounds, ubounds = get_lbounds(BI), get_ubounds(BI)
    n_eqs = get_n_eqs(BI)
    n_ineqs = get_n_ineqs(BI)

    n_evals = 0
    function blackbox(x)
        f = eval_objective(BI, x)
        h = eval_eqs(BI, x)
        g = eval_ineqs(BI, x)
        n_evals += 1
        print_nomad_evaluation(n_evals, [[f] ; h ; g])
        return (true, true, [[f] ; h ; g])
    end

    options = NOMAD.NomadOptions(max_bb_eval = max_evals, display_degree = 0, madspip_optimization = true)
    p = NomadProblem(dimension, 1 + n_eqs + n_ineqs, [["OBJ"] ; ["EQPB" for _ in 1:n_eqs] ; ["PB" for _ in 1:n_ineqs]], blackbox, lower_bound = lbounds, upper_bound = ubounds; options = options)
    return solve(p, x0)
end

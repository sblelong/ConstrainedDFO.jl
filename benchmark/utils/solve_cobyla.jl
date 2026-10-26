using NLPModels
using PRIMA

"""Matrix A and constant term c0 such that the linear constraints of indices idcs read c(x)[idcs] = A x + c0."""
function get_linear_part(nlp::AbstractNLPModel, idcs::Vector{Int})
    # jac_lin and cons_lin have one row per linear constraint, in the order of nlp.meta.lin
    rows = Int.(indexin(idcs, nlp.meta.lin))
    A = Matrix{Float64}(jac_lin(nlp, nlp.meta.x0))[rows, :]
    c0 = cons_lin(nlp, zeros(nlp.meta.nvar))[rows]
    return (A, c0)
end

function solve_cobyla(nlp::AbstractNLPModel, x0::Vector{Float64}; max_evals::Int = 200 * (nlp.meta.nvar + 1))
    # x0 and the functions passed to COBYLA live in the space of the free variables
    R = FixedVarsReduction(nlp)

    lbounds, ubounds = reduce_point(R, nlp.meta.lvar), reduce_point(R, nlp.meta.uvar)

    # Objective function
    objective(x) = obj(nlp, expand_point(R, x))

    lcon, ucon = nlp.meta.lcon, nlp.meta.ucon
    idcs_l = [nlp.meta.jlow ; nlp.meta.jrng] # Indices of constraints of the form c≤g(x)
    idcs_u = [nlp.meta.jupp ; nlp.meta.jrng] # Indices of constraints of the form g(x)≤c

    # Nonlinear constraints, as c_eq(x) = 0 and c_ineq(x) ≤ 0
    nleq_idcs = nlp.meta.nln ∩ nlp.meta.jfix
    nlineq_l_idcs = nlp.meta.nln ∩ idcs_l
    nlineq_u_idcs = nlp.meta.nln ∩ idcs_u
    nonlinear_eq = length(nleq_idcs) > 0 ? x -> cons(nlp, expand_point(R, x))[nleq_idcs] .- ucon[nleq_idcs] : nothing
    function c_ineq(x)
        cons_values = cons(nlp, expand_point(R, x))
        return [lcon[nlineq_l_idcs] .- cons_values[nlineq_l_idcs] ; cons_values[nlineq_u_idcs] .- ucon[nlineq_u_idcs]]
    end
    nonlinear_ineq = length(nlineq_l_idcs) + length(nlineq_u_idcs) > 0 ? c_ineq : nothing

    # Linear constraints, as Aₑ x = bₑ and Aᵢ x ≤ bᵢ
    leq_idcs = nlp.meta.lin ∩ nlp.meta.jfix
    lineq_l_idcs = nlp.meta.lin ∩ idcs_l
    lineq_u_idcs = nlp.meta.lin ∩ idcs_u

    linear_eq = nothing
    if length(leq_idcs) > 0
        A_eq, c0_eq = get_linear_part(nlp, leq_idcs)
        linear_eq = reduce_linear(R, A_eq, ucon[leq_idcs] .- c0_eq)
    end

    linear_ineq = nothing
    if length(lineq_l_idcs) + length(lineq_u_idcs) > 0
        A_l, c0_l = get_linear_part(nlp, lineq_l_idcs)
        A_u, c0_u = get_linear_part(nlp, lineq_u_idcs)
        A_ineq = [-A_l ; A_u]
        b_ineq = [c0_l .- lcon[lineq_l_idcs] ; ucon[lineq_u_idcs] .- c0_u]
        linear_ineq = reduce_linear(R, A_ineq, b_ineq)
    end

    result_cobyla = cobyla(objective, x0; maxfun = max_evals, iprint = PRIMA.MSG_FEVL, xl = lbounds, xu = ubounds, nonlinear_eq = nonlinear_eq, nonlinear_ineq = nonlinear_ineq, linear_eq = linear_eq, linear_ineq = linear_ineq)

    return result_cobyla
end

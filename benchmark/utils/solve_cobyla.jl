using NLPModels
using PRIMA

"""
Need to distinguish linear and nonlinear constraints. Populate the NLPModelsBlackboxInstance interface with functions that do this.
"""

function get_matrix_rhs(nlp::AbstractNLPModel, idcs::Vector{Int})
    x0 = nlp.meta.x0
    A = Matrix{Float64}(jac_lin(nlp, x0))[idcs, :]
    h0 = cons(nlp, zeros(nlp.meta.nvar))[idcs]
    b = -h0
    return (A, b)
end

function solve_cobyla(nlp::AbstractNLPModel, x0::Vector{Float64}; max_evals::Int = 200 * (nlp.meta.nvar + 1))
    lbounds, ubounds = nlp.meta.lvar, nlp.meta.uvar

    objective(x) = obj(nlp, x)

    # Nonlinear constraints
    nleq_idcs = nlp.meta.nln ∩ nlp.meta.jfix
    nlineq_idcs = setdiff(nlp.meta.nln, nleq_idcs)
    nonlinear_eq = length(nleq_idcs) > 0 ? x -> cons(nlp, x)[nleq_idcs] : nothing
    nonlinear_ineq = length(nlineq_idcs) > 0 ? x -> cons(nlp, x)[nlineq_idcs] : nothing

    # Linear constraints
    leq_idcs = nlp.meta.lin ∩ nlp.meta.jfix
    lineq_idcs = setdiff(nlp.meta.lin, leq_idcs)
    linear_eq = length(leq_idcs) > 0 ? get_matrix_rhs(nlp, leq_idcs) : nothing
    linear_ineq = length(lineq_idcs) > 0 ? get_matrix_rhs(nlp, lineq_idcs) : nothing

    result_cobyla = cobyla(objective, x0; maxfun = max_evals, iprint = PRIMA.MSG_FEVL, xl = lbounds, xu = ubounds, nonlinear_eq = nonlinear_eq, nonlinear_ineq = nonlinear_ineq, linear_eq = linear_eq, linear_ineq = linear_ineq)

    return result_cobyla
end

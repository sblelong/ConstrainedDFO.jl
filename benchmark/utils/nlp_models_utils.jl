using NLPModels
using CUTEst
using ConstrainedDFO

function nlp_to_bb(nlp::AbstractNLPModel)
    dimension = nlp.meta.nvar

    f(x) = obj(nlp, x)

    # Constraints
    n_cons = nlp.meta.ncon
    # Equalities
    idcs_eqs = nlp.meta.jfix
    n_eqs = length(idcs_eqs)
    h(x) = cons(nlp, x)[idcs_eqs]

    # Inequalities and bounds altogether
    idcs_ineqs = setdiff(1:n_cons, idcs_eqs)
    n_ineqs = length(idcs_ineqs)
    lbounds(x) = [nlp.meta.lvar[idx] - x[idx] for idx in [nlp.meta.ilow ; nlp.meta.irng]]
    ubounds(x) = [x[idx] - nlp.meta.uvar[idx] for idx in [nlp.meta.iupp ; nlp.meta.irng]]

    g(x) = [-1.0 .* cons(nlp, x)[idcs_ineqs] ; lbounds(x) ; ubounds(x)] # Careful! NLPModels uses g(x)≥0.

    problem = BlackboxProblem(dimension, n_eqs, n_ineqs, f, h, g)

    x0 = nlp.meta.x0

    return BlackboxInstance(problem, x0)
end

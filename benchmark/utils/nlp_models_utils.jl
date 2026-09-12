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
    h(x) = cons(nlp, x)[idcs_eqs] .- nlp.meta.ucon[idcs_eqs]

    # Inequalities and bounds altogether
    idcs_ineqs = setdiff(1:n_cons, idcs_eqs)
    n_ineqs = length(idcs_ineqs)
    function g(x)
        cons_values = cons(nlp, x)
        idcs_ineqs_l = [nlp.meta.jlow ; nlp.meta.jrng] # Indices of constraints of the form c≤g(x)
        lineqs = nlp.meta.lcon[idcs_ineqs_l] .- cons_values[idcs_ineqs_l]
        idcs_ineqs_u = [nlp.meta.jupp ; nlp.meta.jrng]
        uineqs = cons_values[idcs_ineqs_u] .- nlp.meta.lcon[idcs_ineqs_u]
        idcs_lbounds = [nlp.meta.ilow ; nlp.meta.irng]
        lbounds = [nlp.meta.lvar[idx] - x[idx] for idx in idcs_lbounds]
        idcs_ubounds = [nlp.meta.iupp ; nlp.meta.irng]
        ubounds = [x[idx] - nlp.meta.uvar[idx] for idx in idcs_ubounds]
        return [lineqs ; uineqs ; lbounds ; ubounds]
    end

    problem = BlackboxProblem(dimension, n_eqs, n_ineqs, f, h, g)

    x0 = nlp.meta.x0

    return BlackboxInstance(problem, x0)
end

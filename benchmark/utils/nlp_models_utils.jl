using NLPModels
using ConstrainedDFO
using ForwardDiff

"""Defining function carrying the NLPModel needed to differentiate it."""
struct NLPModelEqualityFunction{N, I}
    nlp::N
    indices::I
end

function (h::NLPModelEqualityFunction)(x)
    return cons(h.nlp, x)[h.indices] .- h.nlp.meta.ucon[h.indices]
end

# This method is intentionally defined in the benchmark code: NLPModels
# callbacks are differentiated through NLPModels.jac rather than ForwardDiff.
function ConstrainedDFO.eval_defining_jacobian(M::ConstrainedDFO.EqualityManifold, p)
    h = getfield(M, :defining_function)
    if h isa NLPModelEqualityFunction
        return Matrix(jac(h.nlp, p)[h.indices, :])
    end
    return ForwardDiff.jacobian(h, p)
end

function nlp_to_bb(nlp::AbstractNLPModel)
    dimension = nlp.meta.nvar

    f(x) = obj(nlp, x)

    # Constraints
    n_cons = nlp.meta.ncon

    # Equalities
    idcs_eqs = nlp.meta.jfix
    n_eqs = length(idcs_eqs)
    h = NLPModelEqualityFunction(nlp, idcs_eqs)

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

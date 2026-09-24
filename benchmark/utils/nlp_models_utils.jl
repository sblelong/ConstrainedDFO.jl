using NLPModels
using ConstrainedDFO
using ForwardDiff
using JuMP
using ManifoldsBase

"""Defining function carrying the NLPModel needed to differentiate it."""
struct NLPModelEqualityFunction{N, I}
    nlp::N
    indices::I
end

function (h::NLPModelEqualityFunction)(x)
    return cons(h.nlp, x)[h.indices] .- h.nlp.meta.ucon[h.indices]
end

function (h::NLPModelEqualityFunction)(
        x::AbstractVector{<:JuMP.VariableRef},
    )
    model = JuMP.owner_model(x[1])
    n = length(x)
    expressions = JuMP.NonlinearExpr[]

    for index in h.indices
        name = gensym(:nlp_equality)

        value(args...) = begin
            xx = Float64[args...]
            cons(h.nlp, xx)[index] - h.nlp.meta.ucon[index]
        end

        gradient(g, args...) = begin
            xx = Float64[args...]
            g .= vec(jac(h.nlp, xx)[index, :])
            return
        end

        JuMP.register(model, name, n, value, gradient)

        push!(expressions, JuMP.NonlinearExpr(name, x...))
    end

    return expressions
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

function ConstrainedDFO.eval_defining_hessian(M::ConstrainedDFO.EqualityManifold, p, i::Int)
    h = getfield(M, :defining_function)
    if h isa NLPModelEqualityFunction
        n_eqs = representation_size(M)[1] - manifold_dimension(M)
        idcs_eqs = zeros(n_eqs)
        idcs_eqs[i] = 1.0
        return Matrix(hess(h.nlp, p, idcs_eqs; obj_weight = 0.0))
    end
    hi(x) = h(x)[i]
    return ForwardDiff.hessian(hi, p)
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
        idcs_ineqs_u = [nlp.meta.jupp ; nlp.meta.jrng] # Indices of constraints of the form g(x)≤c
        uineqs = cons_values[idcs_ineqs_u] .- nlp.meta.lcon[idcs_ineqs_u]
        return [lineqs ; uineqs]
    end
    lbounds = nlp.meta.lvar
    ubounds = nlp.meta.uvar

    problem = BlackboxProblem(dimension, n_eqs, n_ineqs, f, h, g; lb = lbounds, ub = ubounds)

    x0 = nlp.meta.x0

    return BlackboxInstance(problem, x0)
end

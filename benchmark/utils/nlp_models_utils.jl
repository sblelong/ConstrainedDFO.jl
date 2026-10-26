using NLPModels
using ConstrainedDFO
using ForwardDiff
using JuMP
using ManifoldsBase
using Ipopt, SparseArrays

"""Defining function carrying the NLPModel needed to differentiate it."""
struct NLPModelEqualityFunction{N, I, F, J, V}
    nlp::N
    indices::I
    free_indices::F
    fixed_indices::J
    fixed_values::V
end

function NLPModelEqualityFunction(nlp, indices)
    free_indices = collect(1:nlp.meta.nvar)
    return NLPModelEqualityFunction(nlp, indices, free_indices, Int[], Float64[])
end

function nlp_full_point(h::NLPModelEqualityFunction, x)
    length(x) == length(h.free_indices) || throw(DimensionMismatch(
        "Expected $(length(h.free_indices)) free variables, got $(length(x))."
    ))
    isempty(h.fixed_indices) && return x

    xf = Vector{eltype(x)}(undef, h.nlp.meta.nvar)
    xf[h.fixed_indices] .= h.fixed_values
    xf[h.free_indices] .= x
    return xf
end

function (h::NLPModelEqualityFunction)(x)
    return cons(h.nlp, nlp_full_point(h, x))[h.indices] .- h.nlp.meta.ucon[h.indices]
end

function (h::NLPModelEqualityFunction)(
        x::AbstractVector{<:JuMP.VariableRef},
    )
    model = JuMP.owner_model(x[1])
    n = length(h.free_indices)
    expressions = JuMP.NonlinearExpr[]

    for index in h.indices
        name = gensym(:nlp_equality)

        value(args...) = begin
            xx = nlp_full_point(h, Float64[args...])
            cons(h.nlp, xx)[index] - h.nlp.meta.ucon[index]
        end

        gradient(g, args...) = begin
            xx = nlp_full_point(h, Float64[args...])
            g .= vec(jac(h.nlp, xx)[index, h.free_indices])
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
        xx = nlp_full_point(h, p)
        return Matrix(jac(h.nlp, xx)[h.indices, h.free_indices])
    end
    return ForwardDiff.jacobian(h, p)
end

function ConstrainedDFO.eval_defining_hessian(M::ConstrainedDFO.EqualityManifold, p, i::Int)
    h = getfield(M, :defining_function)
    if h isa NLPModelEqualityFunction
        nlp = h.nlp
        n_cons = nlp.meta.ncon
        weights_cons = zeros(n_cons)
        weights_cons[h.indices[i]] = 1.0
        xx = nlp_full_point(h, p)
        H = hess(nlp, xx, weights_cons; obj_weight = 0.0)
        return Matrix(H[h.free_indices, h.free_indices])
    end
    hi(x) = h(x)[i]
    return ForwardDiff.hessian(hi, p)
end

function nlp_to_bb(nlp::AbstractNLPModel)
    lbounds_raw, ubounds_raw = nlp.meta.lvar, nlp.meta.uvar
    n_cons = nlp.meta.ncon

    # Objective function
    f_raw(x) = obj(nlp, x)

    # Equalities
    idcs_eqs = nlp.meta.jfix
    n_eqs = length(idcs_eqs)
    h_raw = NLPModelEqualityFunction(nlp, idcs_eqs)

    # Inequalities
    idcs_ineqs = setdiff(1:n_cons, idcs_eqs)
    n_ineqs = length(idcs_ineqs)
    function g_raw(x)
        cons_values = cons(nlp, x)
        idcs_ineqs_l = [nlp.meta.jlow ; nlp.meta.jrng] # Indices of constraints of the form c≤g(x)
        lineqs = nlp.meta.lcon[idcs_ineqs_l] .- cons_values[idcs_ineqs_l]
        idcs_ineqs_u = [nlp.meta.jupp ; nlp.meta.jrng] # Indices of constraints of the form g(x)≤c
        uineqs = cons_values[idcs_ineqs_u] .- nlp.meta.lcon[idcs_ineqs_u]
        return [lineqs ; uineqs]
    end

    idcs_fixed_vars = nlp.meta.ifix
    idcs_other_vars = setdiff(1:nlp.meta.nvar, idcs_fixed_vars)
    has_fixed_vars = length(idcs_fixed_vars) > 0
    h_raw = NLPModelEqualityFunction(
        nlp,
        idcs_eqs,
        idcs_other_vars,
        idcs_fixed_vars,
        lbounds_raw[idcs_fixed_vars],
    )
    function add_fixed_vars(x)
        xf = Vector{Float64}(undef, nlp.meta.nvar)
        xf[idcs_fixed_vars] .= lbounds_raw[idcs_fixed_vars]
        xf[idcs_other_vars] .= x
        return xf
    end

    dimension = nlp.meta.nvar - length(nlp.meta.ifix)

    f(x) = has_fixed_vars ? f_raw(add_fixed_vars(x)) : f_raw(x)
    h(x) = has_fixed_vars ? h_raw(add_fixed_vars(x)) : h_raw(x)
    g(x) = has_fixed_vars ? g_raw(add_fixed_vars(x)) : g_raw(x)

    lbounds = lbounds_raw[idcs_other_vars]
    ubounds = ubounds_raw[idcs_other_vars]

    problem = BlackboxProblem(dimension, n_eqs, n_ineqs, f, h, g; lb = lbounds, ub = ubounds)

    x0 = nlp.meta.x0[idcs_other_vars]

    return BlackboxInstance(problem, x0)
end

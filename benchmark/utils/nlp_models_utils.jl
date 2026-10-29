using NLPModels
using ConstrainedDFO
using ForwardDiff
using JuMP
using ManifoldsBase
using Ipopt, SparseArrays

"""Reduction of a problem to its free variables, the fixed variables being set to their bound value."""
struct FixedVarsReduction
    nvar::Int
    free_indices::Vector{Int}
    fixed_indices::Vector{Int}
    fixed_values::Vector{Float64}
end

"""Reduction that eliminates the fixed variables of an NLPModel."""
function FixedVarsReduction(nlp::AbstractNLPModel)
    fixed_indices = nlp.meta.ifix
    free_indices = setdiff(1:nlp.meta.nvar, fixed_indices)
    return FixedVarsReduction(nlp.meta.nvar, free_indices, fixed_indices, nlp.meta.lvar[fixed_indices])
end

"""Identity reduction: all variables are free."""
FixedVarsReduction(nvar::Int) = FixedVarsReduction(nvar, collect(1:nvar), Int[], Float64[])

has_fixed_vars(R::FixedVarsReduction) = !isempty(R.fixed_indices)
reduced_dimension(R::FixedVarsReduction) = length(R.free_indices)

"""Build the full point from the values of the free variables."""
function expand_point(R::FixedVarsReduction, x)
    length(x) == reduced_dimension(R) || throw(DimensionMismatch(
        "Expected $(reduced_dimension(R)) free variables, got $(length(x))."
    ))
    has_fixed_vars(R) || return x

    xf = Vector{eltype(x)}(undef, R.nvar)
    xf[R.fixed_indices] .= R.fixed_values
    xf[R.free_indices] .= x
    return xf
end

"""Extract the free variables from a full point (also applies to bounds)."""
reduce_point(R::FixedVarsReduction, x) = x[R.free_indices]

"""Rewrite the linear system A x = b (or A x ≤ b) in terms of the free variables only."""
function reduce_linear(R::FixedVarsReduction, A::AbstractMatrix, b::AbstractVector)
    A_free = A[:, R.free_indices]
    b_free = b .- A[:, R.fixed_indices] * R.fixed_values
    return (A_free, b_free)
end

"""Defining function carrying the NLPModel needed to differentiate it."""
struct NLPModelEqualityFunction{N, I}
    nlp::N
    indices::I
    reduction::FixedVarsReduction
end

NLPModelEqualityFunction(nlp, indices) = NLPModelEqualityFunction(nlp, indices, FixedVarsReduction(nlp.meta.nvar))

nlp_full_point(h::NLPModelEqualityFunction, x) = expand_point(h.reduction, x)

function (h::NLPModelEqualityFunction)(x)
    return cons(h.nlp, nlp_full_point(h, x))[h.indices] .- h.nlp.meta.ucon[h.indices]
end

function (h::NLPModelEqualityFunction)(
        x::AbstractVector{<:JuMP.VariableRef},
    )
    model = JuMP.owner_model(x[1])
    n = reduced_dimension(h.reduction)
    expressions = JuMP.NonlinearExpr[]

    for index in h.indices
        name = gensym(:nlp_equality)

        value(args...) = begin
            xx = nlp_full_point(h, Float64[args...])
            cons(h.nlp, xx)[index] - h.nlp.meta.ucon[index]
        end

        gradient(g, args...) = begin
            xx = nlp_full_point(h, Float64[args...])
            g .= vec(jac(h.nlp, xx)[index, h.reduction.free_indices])
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
        return Matrix(jac(h.nlp, xx)[h.indices, h.reduction.free_indices])
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
        return Matrix(H[h.reduction.free_indices, h.reduction.free_indices])
    end
    hi(x) = h(x)[i]
    return ForwardDiff.hessian(hi, p)
end

"""
Blackbox built from an NLPModel, as a function of the free variables.
If `tol_eqs` is given, the equality constraints are unrelaxable: when any of them is violated by more than `tol_eqs`,
the objective and inequality constraints return FAILURE_MAX. The equality constraints and bounds are unaffected.
"""
function nlp_to_bb(nlp::AbstractNLPModel; tol_eqs::Union{Float64, Nothing} = nothing)
    lbounds_raw, ubounds_raw = nlp.meta.lvar, nlp.meta.uvar

    # Objective function
    f_raw(x) = obj(nlp, x)

    # Equalities
    idcs_eqs = nlp.meta.jfix
    n_eqs = length(idcs_eqs)

    # Inequalities
    idcs_ineqs_l = [nlp.meta.jlow ; nlp.meta.jrng] # Indices of constraints of the form c≤g(x)
    idcs_ineqs_u = [nlp.meta.jupp ; nlp.meta.jrng] # Indices of constraints of the form g(x)≤c
    n_ineqs = length(idcs_ineqs_l) + length(idcs_ineqs_u) # Ranged constraints count twice
    function g_raw(x)
        cons_values = cons(nlp, x)
        lineqs = nlp.meta.lcon[idcs_ineqs_l] .- cons_values[idcs_ineqs_l]
        uineqs = cons_values[idcs_ineqs_u] .- nlp.meta.ucon[idcs_ineqs_u]
        return [lineqs ; uineqs]
    end

    R = FixedVarsReduction(nlp)
    # Already takes the free variables as input, and must not be wrapped so that
    # EqualityManifold can dispatch on NLPModelEqualityFunction
    h = NLPModelEqualityFunction(nlp, idcs_eqs, R)

    dimension = reduced_dimension(R)

    eqs_satisfied(x) = isnothing(tol_eqs) || all(abs.(h(x)) .≤ tol_eqs)
    f(x) = eqs_satisfied(x) ? f_raw(expand_point(R, x)) : FAILURE_MAX
    g(x) = eqs_satisfied(x) ? g_raw(expand_point(R, x)) : fill(FAILURE_MAX, n_ineqs)

    lbounds = reduce_point(R, lbounds_raw)
    ubounds = reduce_point(R, ubounds_raw)

    problem = BlackboxProblem(dimension, n_eqs, n_ineqs, f, h, g; lb = lbounds, ub = ubounds)

    x0 = reduce_point(R, nlp.meta.x0)

    return BlackboxInstance(problem, x0)
end

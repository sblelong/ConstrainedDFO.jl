using NLPModels
using ConstrainedDFO
using ForwardDiff
using JuMP
using ManifoldsBase
using Ipopt, SparseArrays

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
        nlp = h.nlp
        n_cons = nlp.meta.ncon
        weights_cons = zeros(n_cons)
        idcs_eqs = nlp.meta.jfix
        weights_cons[idcs_eqs[i]] = 1.0
        return Matrix(hess(nlp, p, weights_cons; obj_weight = 0.0))
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

function make_x0_feasible(nlp::AbstractNLPModel)
    n = nlp.meta.nvar
    eq = nlp.meta.jfix                        # equality constraint indices
    lb = nlp.meta.lvar
    ub = nlp.meta.uvar
    x0 = nlp.meta.x0

    A = jac(nlp, x0)[eq, :]
    c_eval = cons(nlp, x0)               # coefficient matrix (linear, constant)
    b = -c_eval[eq] + A * x0                # RHS (see previous discussion)

    model = Model(Ipopt.Optimizer)
    set_silent(model)
    set_attribute(model, "tol", 1.0e-12)
    set_attribute(model, "constr_viol_tol", 1.0e-10)
    set_attribute(model, "acceptable_constr_viol_tol", 1.0e-10)

    @variable(model, x[i = 1:n])
    for i in 1:n
        isfinite(lb[i]) && set_lower_bound(x[i], lb[i] + 1.0e-8) # Adding a numerical tolerance because NOMAD is picky.
        isfinite(ub[i]) && set_upper_bound(x[i], ub[i] - 1.0e-8)
    end
    @constraint(model, A * x .== b)
    @objective(model, Min, sum((x[i] - x0[i])^2 for i in 1:n))

    optimize!(model)

    if termination_status(model) == LOCALLY_SOLVED
        x_sol = clamp.(value.(x), lb, ub)
        return x_sol
    else
        error("No feasible point exists for this problem.")
    end
end

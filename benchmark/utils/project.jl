using NLPModels
using JuMP, Ipopt
using ConstrainedDFO

function make_x0_feasible(BI::BlackboxInstance; perturbation_attemps::Int = 10)
    n = get_dimension(BI)
    lb = get_lbounds(BI)
    ub = get_ubounds(BI)
    x0 = ConstrainedDFO.get_x0(BI)
    h(x) = eval_eqs(BI, x)
    p = length(h(x0))

    model = Model(Ipopt.Optimizer)
    set_silent(model)
    set_attribute(model, "tol", 1.0e-12)
    set_attribute(model, "constr_viol_tol", 1.0e-12)
    set_attribute(model, "bound_relax_factor", 0.0)
    @variable(model, x[1:n])
    # Keep the point strictly inside the bounds: the solvers project it exactly onto the equality constraints,
    # which moves it slightly and could push it out of the bounds (e.g. log(x) with x ≥ 1e-12 in HIMMELBJ).
    δ = 1.0e-6
    for i in 1:n
        isfinite(lb[i]) && set_lower_bound(x[i], lb[i] + δ * max(1.0, abs(lb[i])))
        isfinite(ub[i]) && set_upper_bound(x[i], ub[i] - δ * max(1.0, abs(ub[i])))
    end
    @NLobjective(model, Min, 0.5 * sum((x[i] - x0[i])^2 for i in 1:n))
    @NLconstraint(model, [j = 1:p], h(x)[j] == 0)

    optimize!(model)

    termination_status(model) ∈ (MOI.OPTIMAL, MOI.LOCALLY_SOLVED) && return value.(x)

    for _ in 1:perturbation_attemps
        set_start_value.(x, x0 .+ 0.1 * randn(n))
        optimize!(model)
        termination_status(model) ∈ (MOI.OPTIMAL, MOI.LOCALLY_SOLVED) && return value.(x)
    end

    error("No feasible first guess found for this problem.")
end

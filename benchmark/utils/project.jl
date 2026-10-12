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
    set_attribute(model, "constr_viol_tol", 1.0e-8)
    @variable(model, x[1:n])
    for i in 1:n
        isfinite(lb[i]) && set_lower_bound(x[i], lb[i])
        isfinite(ub[i]) && set_upper_bound(x[i], ub[i])
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

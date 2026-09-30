using JuMP, Ipopt

function project_on_solution_space(x, A::Matrix{Float64}, b::Vector{Float64})
    dimension = length(x)

    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, y[1:dimension])
    @NLobjective(model, Min, 0.5 * sum((y[i] - x[i])^2 for i in 1:dimension))
    @constraint(model, A * y == b)

    optimize!(model)
    q = value.(y)
    return q
end

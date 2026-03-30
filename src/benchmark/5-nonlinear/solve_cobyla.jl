using ConstrainedDFO
using PRIMA
using JSON
using ProgressBars

data_path = "/home/sblelong/.julia/dev/ConstrainedDFO/src/benchmark/5-nonlinear/data/cobyla/no-process/"
STUPID_MAX = 1.0e20

global x_obj::Vector{Vector{Float64}}
global objective_values_storage::Vector{Float64}
function objective_wrapper(obj::Function, x)
    global objective_values_storage
    global x_obj
    f = obj(x)
    push!(objective_values_storage, f)
    push!(x_obj, x)
    return f
end

global ineqs_values_storage::Vector{Vector{Float64}}
function ineqs_wrapper(ineqs::Function, x)
    global ineqs_values_storage
    g = ineqs(x)
    push!(ineqs_values_storage, g)
    return g
end

global eqs_values_storage::Vector{Vector{Float64}}
global x_eqs::Vector{Vector{Float64}}
function eqs_wrapper(eqs::Function, x)
    global eqs_values_storage
    global x_eqs
    push!(x_eqs, x)
    h = eqs(x)
    push!(eqs_values_storage, h)
    return h
end

# for (i, problem) in ProgressBar(enumerate(nonlinear_benchmarks[5:5]))
problem = nonlinear_benchmarks[5]
i = 1
global objective_values_storage
objective_values_storage = []

global ineqs_values_storage
ineqs_values_storage = Vector{Float64}[]

global eqs_values_storage
eqs_values_storage = Vector{Float64}[]

global x_obj
x_obj = Vector{Float64}[]

global x_eqs
x_eqs = Vector{Float64}[]

n = get_dimension(problem)
f(x) = objective_wrapper(y -> eval_obj(problem, y), x)
x0 = get_x0(problem)
h(x) = eqs_wrapper(y -> eval_eqs(problem, y), x)

data = Dict()

if has_inequality_constraints(problem)
    g(x) = ineqs_wrapper(y -> eval_ineqs(problem, y), x)
    result, solver_status = cobyla(f, x0; maxfun = 1000 * n, iprint = PRIMA.MSG_FEVL, nonlinear_ineq = g, nonlinear_eq = h)
    data["f"] = objective_values_storage
    data["g"] = ineqs_values_storage
    data["h"] = eqs_values_storage
else
    result, solver_status = cobyla(f, x0; maxfun = 1000 * n, iprint = PRIMA.MSG_FEVL, nonlinear_eq = h)
    data["f"] = objective_values_storage
    data["h"] = eqs_values_storage
end

open(joinpath(data_path, "$(i + 4).json"), "w") do io
    JSON.print(io, data)
end
# end

using Pkg
Pkg.activate(".")
using ConstrainedDFO
using DelimitedFiles

id_instance = parse(Int, ARGS[1])
instance = manifold_benchmarks[id_instance]

θ_path = ARGS[2]
θ = readdlm(θ_path)[1, :]
x = spherical_to_cartesian(θ)

f = eval_obj(instance, x)

println("$(f)")

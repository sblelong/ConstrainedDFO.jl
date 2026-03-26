using JSON
using CairoMakie
using LaTeXStrings

# Total number of problems
n_instances = 49
# Total number of algorithms
n_algs = 4

data_dict = Dict(
    "NOverSpectral" => Vector{Float64}[],
    "Exp" => Vector{Float64}[],
    "manopt" => Vector{Float64}[],
    "parametrization" => Vector{Float64}[],
)

base_data_path = "/home/sblelong/.julia/dev/ConstrainedDFO/src/benchmark/4-manifolds/data"

# Maximum number of evaluations will be the largest amount of evaluations used by any algorithm on any problem. So that the data fits into a 3-dimensional matrix.
# Find out about this number.
global max_evaluations::Int
max_evaluations = 0

# For each algorithm: data_dict[alg][i] is the data of algorithm `alg` on problem i.
for (alg_name, alg_data) in data_dict
    # Will adjust the length of the vectors later.
    for id_instance in 1:n_instances
        global max_evaluations
        data_alg_pb = JSON.parsefile(joinpath(base_data_path, "$(alg_name)", "$(id_instance).json"))
        # TODO. Pay attention to the way f_alg_pb is retrieved when using for problems with ineq constraints, or nonfeasible methods.
        f_alg_pb = data_alg_pb["stratified_f"]
        max_evaluations = max(max_evaluations, length(f_alg_pb))
        push!(data_dict[alg_name], f_alg_pb)
    end
end

# Turn this data into a neat 3-dimensional Array.
# Scheme: data_array[id_alg, id_instance, i] = best value at i-th blackbox evaluation for solver id_alg on instance id_instance.
global data_array::Array{Float64, 3}
data_array = fill(typemax(Float64), (n_algs, n_instances, max_evaluations))

for (alg_id, (alg_name, alg_data)) in enumerate(data_dict)
    println("Alg $(alg_id) is $(alg_name)")
    # For every instance
    for id_instance in 1:n_instances
        global data_array

        data_from_dict = data_dict[alg_name][id_instance]

        # How many evals were performed by this solver on this instance?
        n_evals_data = length(data_from_dict)

        # Fill every data vector up to max_evaluations.
        # This approach is only valid because the algorithm already returns stratified fs.
        # TODO pay attention to this when using for other test sets.
        filled_data = fill(typemax(Float64), max_evaluations)

        filled_data[1:n_evals_data] .= data_from_dict
        filled_data[(n_evals_data + 1):end] .= data_from_dict[end]

        data_array[alg_id, id_instance, :] .= filled_data
    end
end

###########################################
# PERFORMANCE PROFILES
###########################################

# Tolerance for resolution
τ = 1.0e-5
# A stupid max for Naps.
STUPID_MAX::Int = 1.0e6

# Find optimals for all problems
optimals = [minimum([data_array[alg_id, id_instance, end] for alg_id in 1:n_algs]) for id_instance in 1:n_instances]

# Compute accuracy ratios at each evaluation for each algorithm on each instance.
accuracy_ratios = zeros((n_algs, n_instances, max_evaluations))
for alg_id in 1:n_algs
    for id_instance in 1:n_instances
        f0 = data_array[alg_id, id_instance, 1]
        for n_eval in 2:max_evaluations
            accuracy_ratios[alg_id, id_instance, n_eval] = (data_array[alg_id, id_instance, n_eval] == f0) ? 0.0 : (data_array[alg_id, id_instance, n_eval] - f0) / (optimals[id_instance] - f0)
        end
    end
end

# Which problems were τ-solved at some point, by each algorithm?
Taps = accuracy_ratios[:, :, end] .≥ 1 - τ

# Find Nap, the first evaluation where each problem was τ-solved (if it was at all).
# Naps[alg_id, id_instance] = index of the first evaluation where instance `id_instance` was τ-solved by solver `alg_id`. If it was not, then put a numerical max.
Naps = fill(typemax(Int), (n_algs, n_instances))
for alg_id in 1:n_algs
    for id_instance in 1:n_instances
        first_solved_index = findfirst(accuracy_ratios[alg_id, id_instance, :] .≥ 1 - τ)
        Naps[alg_id, id_instance] = isnothing(first_solved_index) ? STUPID_MAX : first_solved_index
    end
end

# Compute rap.
# raps[alg_id, id_instance] = Naps[alg_id, id_instance] / min(Naps of all algorithms that have τ-solved the same instance).
raps = zeros(n_algs, n_instances)
for id_instance in 1:n_instances # This outer loop should be about instances, so the minimum can be easily found every time.
    smallest_first_solved_index = minimum(Naps[:, id_instance])
    for alg_id in 1:n_algs
        if Taps[alg_id, id_instance] == 0
            raps[alg_id, id_instance] = STUPID_MAX
        elseif smallest_first_solved_index == STUPID_MAX
            raps[alg_id, id_instance] = STUPID_MAX
        else
            raps[alg_id, id_instance] = Naps[alg_id, id_instance] / smallest_first_solved_index
        end
    end
end

# Compute the performance profile function
αmax::Float64 = 20.0

# Values of the performance profile function
# ρaαs[alg_id, α] = portion of problems for which raps[alg_id, :] ≤ α.
αs = LinRange(1.0, αmax, 200)
ρaαs = zeros(n_algs, length(αs))
for alg_id in 1:n_algs
    for (i, α) in enumerate(αs) # Need to use enumerate here to affect the values of ρ to the table ρaαs.
        ρaαs[alg_id, i] = 1 / n_instances * sum(raps[alg_id, :] .≤ α)
    end
end

# Plot the performance profile function
with_theme(theme_latexfonts()) do
    fig = Figure(size = (600, 400))
    Axis(
        fig[1, 1],
        limits = ((1.0, αmax), (0.0, 1.05)),
        xlabel = L"Ratio $\alpha$ d'évaluations",
        ylabel = L"Proportion $\rho_a(\alpha)$ de problèmes $\tau$-résolus",
        xlabelsize = 20,
        ylabelsize = 20
    )
    stairs!(αs, ρaαs[2, :]; label = L"\text{RDFO} ($\mathrm{proj}_{\mathcal{M}}$)")
    stairs!(αs, ρaαs[4, :]; label = L"\text{RDFO} ($\mathrm{Exp}$)")
    stairs!(αs, ρaαs[3, :]; label = "MADS (Manopt.jl)")
    stairs!(αs, ρaαs[1, :]; label = "Coordonnées sphériques")
    axislegend(position = :rb; labelsize = 20)
    CairoMakie.save("/home/sblelong/msc-thesis/thesis/figures/num-manopt-performance-profile-$(τ).pdf", fig; px_per_unit = 4)
end

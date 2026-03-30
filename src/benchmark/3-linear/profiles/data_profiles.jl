using JSON
using CairoMakie
using LaTeXStrings

# Total number of problems
n_instances = 42
# Total number of algorithms
n_algs = 2

data_dict = Dict(
    "rdfo" => Vector{Float64}[],
    "nomad_converter_svd" => Vector{Float64}[],
)

base_data_path = "/home/sblelong/.julia/dev/ConstrainedDFO/src/benchmark/3-linear/data"

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
# DATA PROFILES
###########################################

# Tolerance for resolution
τ = 1.0e-5
# A stupid max for Naps.
STUPID_MAX::Int = 1.0e6

# Find optimals for all problems
optimals = [minimum([data_array[alg_id, id_instance, end] for alg_id in 1:n_algs]) for id_instance in 1:n_instances]

# Find dimensions of all problems.
# TODO pay attention to this when using for other test sets.
dimensions = [get_dimension(pb) for pb in linear_benchmarks]

# Compute accuracy ratios at each evaluation for each algorithm on each instance.
accuracy_ratios = zeros((n_algs, n_instances, max_evaluations))
for alg_id in 1:n_algs
    for id_instance in 1:n_instances
        f0 = data_array[alg_id, id_instance, 1]
        if f0 == 1.0e20
            # The accuracy ratios should be computed with f0 = objective value for the first feasible iterate found.
            if data_array[alg_id, id_instance, end] == f0
                continue # The accuracy ratios remain 0 for this problem.
            else
                first_feasible_idx = findfirst(x -> x ≠ 1.0e20, data_array[alg_id, id_instance, :])
                first_feasible_obj = data_array[alg_id, id_instance, first_feasible_idx]
                # println("Alg $(alg_id) on instance $(id_instance). First feasible value: $(first_feasible_obj)")
                for n_eval in 2:max_evaluations
                    accuracy_ratios[alg_id, id_instance, n_eval] = (min(data_array[alg_id, id_instance, n_eval] - first_feasible_obj, 0.0)) / (optimals[id_instance] - first_feasible_obj)
                end
            end
        else
            for n_eval in 2:max_evaluations
                accuracy_ratios[alg_id, id_instance, n_eval] = (data_array[alg_id, id_instance, n_eval] == f0) ? 0.0 : (data_array[alg_id, id_instance, n_eval] - f0) / (optimals[id_instance] - f0)
            end
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

# Compute the data profile function
k_max = 100

# Values of the data profile function
# daks[alg_id, k] = portion of problems that were τ-solved by solver `alg_id` within k * (dimension + 1) evaluations.
daks = zeros(n_algs, k_max + 1)
for alg_id in 1:n_algs
    for k in 2:(k_max + 1)
        daks[alg_id, k] = (1 / n_instances) * sum(Naps[alg_id, :] .≤ k .* (dimensions .+ 1) .* Taps[alg_id, :])
    end
end

# Plot the data profile function
with_theme(theme_latexfonts()) do
    fig = Figure(size = (600, 400))
    Axis(
        fig[1, 1],
        limits = ((0, k_max), (0.0, 1.05)),
        xlabel = L"Groupes de $n+1$ évaluations",
        ylabel = L"Proportion de problèmes $\tau$-résolus",
        xlabelsize = 22,
        ylabelsize = 22
    )
    stairs!(0:k_max, daks[1, :]; label = L"\text{RDFO}", linewidth = 2)
    stairs!(0:k_max, daks[2, :]; label = L"\text{MADS avec convertisseur linéaire}", linewidth = 2)
    axislegend(position = :rb; labelsize = 25)
    CairoMakie.save("/home/sblelong/msc-thesis/thesis/figures/num-linear-data-profile-$(τ).pdf", fig; px_per_unit = 4)
end

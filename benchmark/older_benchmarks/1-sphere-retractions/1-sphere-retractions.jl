using ConstrainedDFO
using Manifolds

A2 = [
    3.0 -6.5 ;
    -6.5 5.0
]

A3 = [
    -8.0 -0.5 -0.5 ;
    -0.5 -9.0 3.0 ;
    -0.5 3.0 4.0
]

A5 = [
    -6.0 1.5 0.5 0.5 4.5 ;
    1.5 -7.0 -6.0 -1.0 -2.5 ;
    0.5 -6.0 -4.0 1.5 -3.0 ;
    0.5 -1.0 1.5 -10.0 1.0 ;
    4.5 -2.5 -3.0 1.0 -9.0
]

A7 = [
    -3.0   3.0   6.0   7.5   4.5  -5.0  -2.0 ;
    3.0  -7.0   2.0  -5.5   9.0   1.5   3.0 ;
    6.0   2.0  -3.0   3.0   1.0   7.5  -4.5 ;
    7.5  -5.5   3.0  10.0  -4.0   4.5  -8.5 ;
    4.5   9.0   1.0  -4.0   4.0  -5.5   6.0 ;
    -5.0   1.5   7.5   4.5  -5.5   7.0   5.0 ;
    -2.0   3.0  -4.5  -8.5   6.0   5.0   9.0
]

instances_1_eq = [
    BenchmarkEqualityProblem(2, obj_axis, x -> [], x -> [eq_sphere(x)], [-1.0, 0.0]),
    BenchmarkEqualityProblem(3, obj_axis, x -> [], x -> [eq_sphere(x)], (sqrt(3) / 3) .* ones(3)),
    BenchmarkEqualityProblem(5, obj_axis, x -> [], x -> [eq_sphere(x)], [-1.0, 0.0, 0.0, 0.0, 0.0]),
    BenchmarkEqualityProblem(7, obj_axis, x -> [], x -> [eq_sphere(x)], -(sqrt(7) / 7) .* ones(7)),

    BenchmarkEqualityProblem(2, obj_lin, x -> [], x -> [eq_sphere(x)], [1.0, 0.0]),
    BenchmarkEqualityProblem(3, obj_lin, x -> [], x -> [eq_sphere(x)], (sqrt(3) / 3) .* ones(3)),
    BenchmarkEqualityProblem(5, obj_lin, x -> [], x -> [eq_sphere(x)], [1.0, 0.0, 0.0, 0.0, 0.0]),
    BenchmarkEqualityProblem(7, obj_lin, x -> [], x -> [eq_sphere(x)], -(sqrt(7) / 7) .* ones(7)),

    BenchmarkEqualityProblem(2, obj_prod, x -> [], x -> [eq_sphere(x)], [1.0, 0.0]),
    BenchmarkEqualityProblem(3, obj_prod, x -> [], x -> [eq_sphere(x)], (sqrt(3) / 3) .* ones(3)),
    BenchmarkEqualityProblem(5, obj_prod, x -> [], x -> [eq_sphere(x)], [1.0, 0.0, 0.0, 0.0, 0.0]),
    BenchmarkEqualityProblem(7, obj_prod, x -> [], x -> [eq_sphere(x)], (sqrt(7) / 7) .* ones(7)),
    BenchmarkEqualityProblem(10, obj_prod, x -> [], x -> [eq_sphere(x)], -(sqrt(10) / 10) .* ones(10)),
    BenchmarkEqualityProblem(12, obj_prod, x -> [], x -> [eq_sphere(x)], -(sqrt(12) / 12) .* ones(12)),

    BenchmarkEqualityProblem(2, obj_spower, x -> [], x -> [eq_sphere(x)], [1.0, 0.0]),
    BenchmarkEqualityProblem(3, obj_spower, x -> [], x -> [eq_sphere(x)], (sqrt(3) / 3) .* ones(3)),
    BenchmarkEqualityProblem(5, obj_spower, x -> [], x -> [eq_sphere(x)], [1.0, 0.0, 0.0, 0.0, 0.0]),
    BenchmarkEqualityProblem(7, obj_spower, x -> [], x -> [eq_sphere(x)], -(sqrt(7) / 7) .* ones(7)),
    BenchmarkEqualityProblem(10, obj_spower, x -> [], x -> [eq_sphere(x)], -(sqrt(10) / 10) .* ones(10)),
    BenchmarkEqualityProblem(12, obj_spower, x -> [], x -> [eq_sphere(x)], -(sqrt(12) / 12) .* ones(12)),

    BenchmarkEqualityProblem(2, x -> obj_rayleigh(x, A2), x -> [], x -> [eq_sphere(x)], [1.0, 0.0]),
    BenchmarkEqualityProblem(3, x -> obj_rayleigh(x, A3), x -> [], x -> [eq_sphere(x)], (sqrt(3) / 3) .* ones(3)),
    BenchmarkEqualityProblem(5, x -> obj_rayleigh(x, A5), x -> [], x -> [eq_sphere(x)], [1.0, 0.0, 0.0, 0.0, 0.0]),
    BenchmarkEqualityProblem(7, x -> obj_rayleigh(x, A7), x -> [], x -> [eq_sphere(x)], -(sqrt(7) / 7) .* ones(7)),

    BenchmarkEqualityProblem(6, obj_hs54, x -> [], x -> [eq_sphere(x)], (sqrt(6) / 6) .* ones(6)),
]

instances_1_man = [
    BenchmarkManifoldProblem(Manifolds.Sphere(1), obj_axis, x -> [], [-1.0, 0.0]),
    BenchmarkManifoldProblem(Manifolds.Sphere(2), obj_axis, x -> [], (sqrt(3) / 3) .* ones(3)),
    BenchmarkManifoldProblem(Manifolds.Sphere(4), obj_axis, x -> [], [-1.0, 0.0, 0.0, 0.0, 0.0]),
    BenchmarkManifoldProblem(Manifolds.Sphere(6), obj_axis, x -> [], -(sqrt(7) / 7) .* ones(7)),

    BenchmarkManifoldProblem(Manifolds.Sphere(1), obj_lin, x -> [], [1.0, 0.0]),
    BenchmarkManifoldProblem(Manifolds.Sphere(2), obj_lin, x -> [], (sqrt(3) / 3) .* ones(3)),
    BenchmarkManifoldProblem(Manifolds.Sphere(4), obj_lin, x -> [], [1.0, 0.0, 0.0, 0.0, 0.0]),
    BenchmarkManifoldProblem(Manifolds.Sphere(6), obj_lin, x -> [], -(sqrt(7) / 7) .* ones(7)),

    BenchmarkManifoldProblem(Manifolds.Sphere(1), obj_prod, x -> [], [1.0, 0.0]),
    BenchmarkManifoldProblem(Manifolds.Sphere(2), obj_prod, x -> [], (sqrt(3) / 3) .* ones(3)),
    BenchmarkManifoldProblem(Manifolds.Sphere(4), obj_prod, x -> [], [1.0, 0.0, 0.0, 0.0, 0.0]),
    BenchmarkManifoldProblem(Manifolds.Sphere(6), obj_prod, x -> [], (sqrt(7) / 7) .* ones(7)),
    BenchmarkManifoldProblem(Manifolds.Sphere(9), obj_prod, x -> [], -(sqrt(10) / 10) .* ones(10)),
    BenchmarkManifoldProblem(Manifolds.Sphere(11), obj_prod, x -> [], -(sqrt(12) / 12) .* ones(12)),

    BenchmarkManifoldProblem(Manifolds.Sphere(1), obj_spower, x -> [], [1.0, 0.0]),
    BenchmarkManifoldProblem(Manifolds.Sphere(2), obj_spower, x -> [], (sqrt(3) / 3) .* ones(3)),
    BenchmarkManifoldProblem(Manifolds.Sphere(4), obj_spower, x -> [], [1.0, 0.0, 0.0, 0.0, 0.0]),
    BenchmarkManifoldProblem(Manifolds.Sphere(6), obj_spower, x -> [], -(sqrt(7) / 7) .* ones(7)),
    BenchmarkManifoldProblem(Manifolds.Sphere(9), obj_spower, x -> [], -(sqrt(10) / 10) .* ones(10)),
    BenchmarkManifoldProblem(Manifolds.Sphere(11), obj_spower, x -> [], -(sqrt(12) / 12) .* ones(12)),

    BenchmarkManifoldProblem(Manifolds.Sphere(1), x -> obj_rayleigh(x, A2), x -> [], [1.0, 0.0]),
    BenchmarkManifoldProblem(Manifolds.Sphere(2), x -> obj_rayleigh(x, A3), x -> [], (sqrt(3) / 3) .* ones(3)),
    BenchmarkManifoldProblem(Manifolds.Sphere(4), x -> obj_rayleigh(x, A5), x -> [], [1.0, 0.0, 0.0, 0.0, 0.0]),
    BenchmarkManifoldProblem(Manifolds.Sphere(6), x -> obj_rayleigh(x, A7), x -> [], -(sqrt(7) / 7) .* ones(7)),

    BenchmarkManifoldProblem(Manifolds.Sphere(5), obj_hs54, x -> [], -(sqrt(6) / 6) .* ones(6)),
]

n_instances = length(instances_1_eq)

# Solve problems with EqualityManifold structure, so with the projection retraction.

max_evaluations = 2400

one_spec_history = Vector{Float64}[]

print("Solving $(length(instances_1_eq)) problems with the EqualityManifold structure and bound OneOverSpectral... ")

for instance in instances_1_eq
    res, ed, ih, oh, vs = solve_problem(RDFOSolver(), instance; invertibility_bound = OneOverSpectral())
    cat_oh = reduce(vcat, (oh[i] for i in 1:length(oh)))
    strat_data = fill(typemax(Float64), max_evaluations)
    best = typemax(Float64)
    for (i, f_val) in enumerate(cat_oh)
        if f_val < best
            strat_data[i:end] .= f_val
            best = f_val
        end
    end
    push!(one_spec_history, strat_data)
end

println("✓")

print("Solving $(length(instances_1_eq)) problems with the EqualityManifold structure and bound NOverSpectral... ")

n_spec_history = Vector{Float64}[]

for instance in instances_1_eq
    res, ed, ih, oh, vs = solve_problem(RDFOSolver(), instance; invertibility_bound = NOverSpectral())
    cat_oh = reduce(vcat, (oh[i] for i in 1:length(oh)))
    strat_data = fill(typemax(Float64), max_evaluations)
    best = typemax(Float64)
    for (i, f_val) in enumerate(cat_oh)
        if f_val < best
            strat_data[i:end] .= f_val
            best = f_val
        end
    end
    push!(n_spec_history, strat_data)
end

println("✓")

print("Solving $(length(instances_1_eq)) problems with the EqualityManifold structure and bound OneOverSqrtSpectral... ")

one_sqrt_spec_history = Vector{Float64}[]

for instance in instances_1_eq
    res, ed, ih, oh, vs = solve_problem(RDFOSolver(), instance; invertibility_bound = OneOverSqrtSpectral())
    cat_oh = reduce(vcat, (oh[i] for i in 1:length(oh)))
    strat_data = fill(typemax(Float64), max_evaluations)
    best = typemax(Float64)
    for (i, f_val) in enumerate(cat_oh)
        if f_val < best
            strat_data[i:end] .= f_val
            best = f_val
        end
    end
    push!(one_sqrt_spec_history, strat_data)
end

println("✓")

print("Solving $(length(instances_1_eq)) problems with the EqualityManifold structure and bound NOverSqrtSpectral... ")

n_sqrt_spec_history = Vector{Float64}[]

for instance in instances_1_eq
    res, ed, ih, oh, vs = solve_problem(RDFOSolver(), instance; invertibility_bound = NOverSqrtSpectral())
    cat_oh = reduce(vcat, (oh[i] for i in 1:length(oh)))
    strat_data = fill(typemax(Float64), max_evaluations)
    best = typemax(Float64)
    for (i, f_val) in enumerate(cat_oh)
        if f_val < best
            strat_data[i:end] .= f_val
            best = f_val
        end
    end
    push!(n_sqrt_spec_history, strat_data)
end

println("✓")

# Solve problems with the Manifolds.Sphere structure, so with the exponential map as a retraction.

print("Solving $(length(instances_1_man)) problems with the Sphere structure... ")
man_history = Vector{Float64}[]

for instance in instances_1_man
    res, ed, ih, oh, vs = solve_problem(RDFOSolver(), instance)
    cat_oh = reduce(vcat, (oh[i] for i in 1:length(oh)))
    strat_data = fill(typemax(Float64), max_evaluations)
    best = typemax(Float64)
    for (i, f_val) in enumerate(cat_oh)
        if f_val < best
            strat_data[i:end] .= f_val
            best = f_val
        end
    end
    push!(man_history, strat_data)
end

println("✓")

###############################################
# DATA PROFILES
###############################################

STUPID_MAX = 1.0e4

# Turn the data into a 3-dimensional tensor
data = fill(typemax(Float64), (max_evaluations, n_instances, 5))

for instance in 1:n_instances
    data[:, instance, 1] .= one_spec_history[instance]
    data[:, instance, 2] .= n_spec_history[instance]
    data[:, instance, 3] .= one_sqrt_spec_history[instance]
    data[:, instance, 4] .= n_sqrt_spec_history[instance]
    data[:, instance, 5] .= man_history[instance]
end

using CairoMakie
using LaTeXStrings

for τ in [1.0e-2, 1.0e-3, 1.0e-5]
    optimals = [minimum([data[end, instance, i] for i in 1:5]) for instance in 1:n_instances]
    dimensions = [get_dimension(pb) for pb in instances_1_eq]

    # Compute accuracy ratios
    accuracy_ratios = zeros((max_evaluations, n_instances, 5))
    for alg in 1:5
        for instance in 1:n_instances
            f0 = data[1, instance, alg]
            for eval in 2:max_evaluations
                accuracy_ratios[eval, instance, alg] = (data[eval, instance, alg] == f0) ? 0.0 : (data[eval, instance, alg] - f0) / (optimals[instance] - f0)
            end
        end
    end

    # Figure out which problems were solved by each algorithm
    Taps = accuracy_ratios[end, :, :] .≥ 1 - τ

    # Find after how many evaluations each problem was τ-solved
    Naps = fill(typemax(Float64), (n_instances, 5))

    for alg in 1:5
        for instance in 1:n_instances
            idx = findfirst(accuracy_ratios[:, instance, alg] .≥ 1 - τ)
            Naps[instance, alg] = isnothing(idx) ? STUPID_MAX : idx
        end
    end

    k_max = 200
    daks = zeros(k_max + 1, 5)
    for alg in 1:5
        for k in 2:(k_max + 1)
            daks[k, alg] = (1 / n_instances) * sum(Naps[:, alg] .≤ k .* (dimensions .+ 1) .* Taps[:, alg])
        end
    end

    with_theme(theme_latexfonts()) do
        fig = Figure(size = (750, 500))
        Axis(
            fig[1, 1],
            xlabel = L"Groups of $(n_p+1)$ evaluations",
            ylabel = L"Proportion of problems solved $d_a(k)$"
        )
        stairs!(0:200, daks[:, 1]; label = L"\rho(x)=\frac{1}{\lambda(\nabla^2h(x))}")
        stairs!(0:200, daks[:, 2]; label = L"\rho(x)=\frac{n}{\lambda(\nabla^2h(x))}")
        stairs!(0:200, daks[:, 3]; label = L"\rho(x)=\frac{1}{\sqrt{\lambda(\nabla^2h(x))}}")
        stairs!(0:200, daks[:, 4]; label = L"\rho(x)=\frac{n}{\sqrt{\lambda(\nabla^2h(x))}}")
        stairs!(0:200, daks[:, 5]; label = L"\mathrm{Exp}")
        axislegend(position = :rb)
        save("local/figs/1-sphere-retractions/data-profile-$(τ).pdf", fig; px_per_unit = 4)
    end

    println("Done generating the data profile for τ = $(τ)")
end

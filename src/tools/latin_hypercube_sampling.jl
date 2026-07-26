using Random
using ResumableFunctions

"""
    latin_hypercube_sampling(n, lb, ub, p)

Perform a latin hypercube sampling with given bounds, and return `p` sampled points. It is assumed that bounds are well-defined.
"""
function latin_hypercube_sampling(n::Int, lb::Vector{Float64}, ub::Vector{Float64}, p::Int; seed::Union{Int, Nothing} = nothing)::Vector{Vector{Float64}}
    # If one of the variables is not bounded, throw error
    # if (-INFTY in lb) || (INFTY in ub)
    #     throw(ArgumentError("Cannot perform LHS: one of the variables is not bounded."))
    # end

    if !isnothing(seed)
        Random.seed!(seed)
    end

    sample = Vector{Float64}[]

    Π = zeros(Int, p, n)
    for j in 1:n
        Π[:, j] = randperm(p)
    end

    for i in 1:p
        r = rand(n)
        x = lb .+ ((Π[i, :] .- r) ./ p) .* (ub .- lb)
        push!(sample, x)
    end

    return sample
end

export latin_hypercube_sampling

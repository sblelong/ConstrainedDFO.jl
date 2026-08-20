function spherical_to_cartesian(θ::Vector{Float64})
    n = length(θ) + 1
    x = zeros(n)

    x[1] = cos(θ[1])
    for i in 2:(n - 1)
        x[i] = prod([sin(θ[j]) for j in 1:(i - 1)]) * cos(θ[i])
    end
    x[n] = prod(sin.(θ))

    return x
end

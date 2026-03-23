using ManifoldsBase, Manifolds
using ManifoldsBase: ℝ
using LinearAlgebra

import ManifoldsBase: representation_size, manifold_dimension, check_point, check_vector, retract_project!, get_embedding, default_retraction_method, check_size, default_basis, get_basis, get_basis_orthonormal, get_vector_orthonormal!, get_coordinates_orthonormal, exp!

struct ScaledSphere <: AbstractManifold{ℝ}
    dimension::Int
    radius::Float64
end

# Basic setup

manifold_dimension(M::ScaledSphere) = M.dimension

representation_size(M::ScaledSphere) = (M.dimension + 1,)

get_embedding(M::ScaledSphere) = Euclidean(representation_size(M)...)

get_radius(M::ScaledSphere) = M.radius

function check_point(M::ScaledSphere, p; kwargs...)
    if !isapprox(norm(p), M.radius; kwargs...)
        return DomainError(norm(p), "The norm of $p is not $(M.radius).")
    end
    return nothing
end

# Tangent spaces

function check_vector(M::ScaledSphere, p, X; kwargs...)
    if !isapprox(dot(p, X), 0.0; kwargs...)
        return DomainError(
            dot(p, X),
            "The tangent vector $X is not orthogonal to $p."
        )
    end
    return nothing
end

default_basis(::ScaledSphere) = DefaultOrthonormalBasis()

# get_basis(M::ScaledSphere, p, B::DefaultOrthonormalBasis)

function get_basis_orthonormal(M::ScaledSphere, p, N::AbstractNumbers; kwargs...)
    n = manifold_dimension(M)
    r = get_radius(M)
    S = Manifolds.Sphere(n)
    return get_basis_orthonormal(S, p ./ r, N; kwargs)
end

function get_vector_orthonormal!(M::ScaledSphere, Y, p, c, N::AbstractNumbers)
    n = manifold_dimension(M)
    r = get_radius(M)
    S = Manifolds.Sphere(n)
    return get_vector_orthonormal!(S, Y, p ./ r, c, N)
end

# Exponential map and retractions

default_retraction_method(::ScaledSphere) = ExponentialRetraction()

function exp!(M::ScaledSphere, q, p, X)
    n = manifold_dimension(M)
    println(n)
    S = Manifolds.Sphere(n)
    r = get_radius(M)
    q = r .* exp(S, p ./ r, X)
    println(q)
    return q
end

default_invertibility_bound(M::ScaledSphere, m::ExponentialRetraction) = ExactInvertibility()
injectivity_radius(M::ScaledSphere, p, m::ExponentialRetraction) = π * get_radius(M)
invertibility_radius(M::ScaledSphere, p, m::ExponentialRetraction, b::ExactInvertibility) = injectivity_radius(M, p, m)

# Projection

function ManifoldsBase.project(M::ScaledSphere, p)
    r = get_radius(M)
    return r .* (p / norm(p))
end

export ScaledSphere

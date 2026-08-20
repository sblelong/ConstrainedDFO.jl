struct ScaledSphere <: AbstractManifold{ℝ}
    dimension::Int
    radius::Float64
end

# Basic setup

manifold_dimension(M::ScaledSphere) = M.dimension

representation_size(M::ScaledSphere) = (M.dimension + 1,)

get_embedding(M::ScaledSphere) = Euclidean(representation_size(M)...)

get_radius(M::ScaledSphere) = M.radius
get_radius(M::Manifolds.Sphere) = 1.0

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
    nX = norm(X)
    r = get_radius(M)
    if nX == 0
        q .= p
    else
        q .= cos(nX / r) .* p + r * sin(nX / M.radius) .* (1 / nX) .* X
    end
    q .= project(M, q)
    return q
end

default_invertibility_bound(M::ScaledSphere, m::ExponentialRetraction) = ExactInvertibility()
ManifoldsBase.injectivity_radius(M::ScaledSphere) = π * get_radius(M)
ManifoldsBase.injectivity_radius(M::ScaledSphere, p, m::ExponentialRetraction) = π * get_radius(M)
invertibility_radius(M::ScaledSphere, p, m::ExponentialRetraction, b::ExactInvertibility) = injectivity_radius(M, p, m)

# Logarithmic map
function ManifoldsBase.log!(M::ScaledSphere, X, p, q)
    r = get_radius(M)

    cosθ = clamp(dot(p, q) / (r^2), -1, 1)
    return if cosθ ≈ -1
        fill!(X, zero(eltype(X)))
        if p[1] ≈ r
            X[2] = 1
        else
            X[1] = 1
        end
        copyto!(X, X .- dot(p, X) / r .* p)
        X .*= π * r / norm(X)
    else
        θ = acos(cosθ)
        X .= θ .* (q .- cosθ .* p) ./ sin(θ)
    end
    return project!(M, X, p, X)
end

ManifoldsBase.inner(M::ScaledSphere, p, X, Y) = dot(X, Y)

function ManifoldsBase.parallel_transport_to!(M::ScaledSphere, Y, p, X, q)
    n = manifold_dimension(M)
    r = get_radius(M)
    S = Manifolds.Sphere(n)

    Y .= ManifoldsBase.parallel_transport_to(S, p ./ r, X, q ./ r)
    return Y
end

# Projection

function ManifoldsBase.project(M::ScaledSphere, p)
    r = get_radius(M)
    return r .* (p / norm(p))
end

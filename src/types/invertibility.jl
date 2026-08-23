"""
    AbstractInvertibilityBound

A formula to compute a lower bound on the [`invertibility_radius`](@ref) of a manifold.
"""
abstract type AbstractInvertibilityBound end

"""
    ExactInvertibility <: AbstractInvertibilityBound

Computes the exact value of the [`invertibility_radius`](@ref) of a retraction (if possible).
"""
mutable struct ExactInvertibility <: AbstractInvertibilityBound end

"""
    OneOverSpectral <: AbstractInvertibilityBound

Computes a lower bound on the [`invertibility_radius`](@ref) of the [`ProjectionRetraction`](@extref ManifoldsBase.ProjectionRetraction) as

```math
    \\frac{1}{\\max\\{\\lambda(H_{h_i}(x)) : i\\in\\{1,...,p\\}\\}}.
```

with ``\\lambda(H_{h_i})`` the spectral radius of the Hessian matrix of the defining subfunction ``h_i`` for `M`.
"""
mutable struct OneOverSpectral <: AbstractInvertibilityBound end

"""
    NOverSpectral <: AbstractInvertibilityBound

Computes a lower bound on the [`invertibility_radius`](@ref) of the [`ProjectionRetraction`](@extref ManifoldsBase.ProjectionRetraction) as

```math
    \\frac{n}{\\max\\{\\lambda(H_{h_i}(x)) : i\\in\\{1,...,p\\}\\}}.
```

with ``\\lambda(H_{h_i})`` the spectral radius of the Hessian matrix of the defining subfunction ``h_i`` for `M`.
"""
mutable struct NOverSpectral <: AbstractInvertibilityBound end

"""
    OneOverSqrtSpectral <: AbstractInvertibilityBound

Computes a lower bound on the [`invertibility_radius`](@ref) of the [`ProjectionRetraction`](@extref ManifoldsBase.ProjectionRetraction) as

```math
    \\frac{1}{\\sqrt{\\max\\{\\lambda(H_{h_i}(x)) : i\\in\\{1,...,p\\}\\}}}.
```

with ``\\lambda(H_{h_i})`` the spectral radius of the Hessian matrix of the defining subfunction ``h_i`` for `M`.
"""
mutable struct OneOverSqrtSpectral <: AbstractInvertibilityBound end

"""
    NOverSqrtSpectral <: AbstractInvertibilityBound

Computes a lower bound on the [`invertibility_radius`](@ref) of the [`ProjectionRetraction`](@extref ManifoldsBase.ProjectionRetraction) as

```math
    \\frac{n}{\\sqrt{\\max\\{\\lambda(H_{h_i}(x)) : i\\in\\{1,...,p\\}\\}}}.
```

with ``\\lambda(H_{h_i})`` the spectral radius of the Hessian matrix of the defining subfunction ``h_i`` for `M`.
"""
mutable struct NOverSqrtSpectral <: AbstractInvertibilityBound end

"""
    default_invertibility_bound(M::AbstractManifold; m::AbstractRetractionMethod)

Return the default [`AbstractInvertibilityBound`](@ref) used to compute the [`invertibility_radius`](@ref) of ``M`` when endowed with the retraction method `m`. For an [`EqualityManifold`](@ref) endowed with the [`ProjectionRetraction`](@extref ManifoldsBase.ProjectionRetraction), defaults to [`NOverSpectral`](@ref).
"""
function default_invertibility_bound(M::AbstractManifold; m::AbstractRetractionMethod = default_retraction_method(M))
    return _default_invertibility_bound(M, m)
end
_default_invertibility_bound(M::AbstractManifold, m::AbstractRetractionMethod) = ExactInvertibility()
_default_invertibility_bound(M::EqualityManifold, m::ProjectionRetraction) = NOverSpectral()

"""
    invertibility_radius(M::AbstractManifold, p; m::AbstractRetractionMethod, ρ::AbstractInvertibilityBound)

When the manifold ``M`` is endowed with retraction method `m` at `p`, its invertibility radius is defined as

```math
    \\mathrm{inv}(p)=\\sup\\{\\delta>0\\; :\\; R_p\\text{ is a diffeomorphism from }B_p(0;\\delta)\\text{ onto its image}\\}.
```

This function returns a **lower bound** on this quantity, computed according to `ρ`. If `ρ` is an [`ExactInvertibility`](@ref), the exact value is returned.

When `m` is the [`ExponentialRetraction`](@extref ManifoldsBase.ExponentialRetraction), this function falls back to the [`injectivity_radius`](@extref ManifoldsBase.injectivity_radius) of ``M`` at `p`.
"""
function invertibility_radius(M::AbstractManifold, p; m::AbstractRetractionMethod = default_retraction_method(M), ρ::AbstractInvertibilityBound = default_invertibility_bound(M; m = default_retraction_method(M)))
    return _invertibility_radius(M, p, m, ρ)
end

_invertibility_radius(M::AbstractManifold, p, m::ExponentialRetraction, ρ::ExactInvertibility) = injectivity_radius(M, p)
_invertibility_radius(M::Sphere, p, m::StabilizedRetraction, ρ::ExactInvertibility) = injectivity_radius(M, p)

function _invertibility_radius(M::EqualityManifold, p, m::ProjectionRetraction, ρ::OneOverSpectral)
    Hhis = eval_defining_hessians(M, p)
    Λ = [maximum(abs, eigvals(Hhi)) for Hhi in Hhis]
    return 1 / maximum(Λ)
end

function _invertibility_radius(M::EqualityManifold, p, m::ProjectionRetraction, ρ::NOverSpectral)
    n = representation_size(M)[1]
    Hhis = eval_defining_hessians(M, p)
    Λ = [maximum(abs, eigvals(Hhi)) for Hhi in Hhis]
    return n / maximum(Λ)
end

function _invertibility_radius(M::EqualityManifold, p, m::ProjectionRetraction, ρ::OneOverSqrtSpectral)
    Hhis = eval_defining_hessians(M, p)
    Λ = [maximum(abs, eigvals(Hhi)) for Hhi in Hhis]
    return 1 / sqrt(maximum(Λ))
end

function _invertibility_radius(M::EqualityManifold, p, m::ProjectionRetraction, ρ::NOverSqrtSpectral)
    n = representation_size(M)[1]
    Hhis = eval_defining_hessians(M, p)
    Λ = [maximum(abs, eigvals(Hhi)) for Hhi in Hhis]
    return n / sqrt(maximum(Λ))
end

"""
This file contains the description of a submanifold of ℝ^n defined by a unique defining function.
This structure is used to represent the feasible set for equality-constrained problems when such set is a Riemannian submanifold embedded in ℝ^n.
"""

import ManifoldsBase:
    check_size,
    check_point,
    check_vector,
    default_basis,
    default_retraction_method,
    get_basis,
    get_basis_orthonormal,
    get_coordinates_orthonormal,
    get_vector_orthonormal!,
    get_embedding,
    manifold_dimension,
    representation_size,
    retract_project!

using ManifoldsBase
using Manifolds
using LinearAlgebra
using JuMP
using Ipopt
using ForwardDiff

"""
    EqualityManifold <: AbstractManifold{ℝ}

A smooth Riemannian submanifold of ``\\mathbb{R}^n`` defined as the set

```math
    \\mathcal{M}=\\left\\{x\\in\\mathbb{R}^n : h(x)=0\\right\\}
```
for some smooth function ``h: \\mathbb{R}^n\\to\\mathbb{R}`` such that ``\\nabla h(x)`` has full rank for all ``x\\in\\mathcal{M}``.

# Fields

* `defining_function`: the function ``h`` as described above.
* `dimension`: the dimension of the manifold, defined as the common dimension of its tangent spaces.
"""
struct EqualityManifold <: AbstractManifold{ℝ}
    defining_function::Function
    dimension::Int
    embedding_dimension::Int
end

####################################################################
# Some basic features/getters/setters of an `EqualityManifold`.
####################################################################

manifold_dimension(M::EqualityManifold) = M.dimension

representation_size(M::EqualityManifold) = (M.embedding_dimension,)

function get_embedding(M::EqualityManifold)
    return Euclidean(representation_size(M)...)
end

eval_defining_function(M::EqualityManifold, p) = M.defining_function(p)

function eval_defining_jacobian(M::EqualityManifold, p)
    h(x) = eval_defining_function(M, x)
    ∇hp = ForwardDiff.jacobian(h, p)
    return ∇hp
end

function eval_defining_hessian(M::EqualityManifold, p, i::Int)
    hi(x) = eval_defining_function(M, x)[i]
    Hhip = ForwardDiff.hessian(hi, p)
    return Hhip
end

function eval_defining_hessians(M::EqualityManifold, p)
    nb_defining_functions = length(eval_defining_function(M, p))
    hessians = Matrix[]
    for i in 1:nb_defining_functions
        push!(hessians, eval_defining_hessian(M, p, i))
    end
    return hessians
end

####################################################################
# Checks on an `EqualityManifold` and its tangent spaces.
####################################################################

"""
    check_size(M::EqualityManifold, p)

Checks whether point ``p`` has the same length as the `representation_size` of ``M``.
"""
function check_size(M::EqualityManifold, p)
    if size(p) ≠ representation_size(M)
        return DomainError("Vector $(p) cannot belong to $(M) with representation size $(representation_size(M)): it has length $(length(p)).")
    else
        return nothing
    end
end

"""
    check_size(M::EqualityManifold, p, X)

Checks whether point ``p`` and vector ``X`` have the same length as the `representation_size` of ``M``.
"""
function check_size(M::EqualityManifold, p, X)
    if size(X) ≠ representation_size(M)
        return DomainError("Vector $(X) cannot be a tangent vector to $(M) with dimension $(manifold_dimension(M)): it has length $(length(X)).")
    else
        return check_size(M, p)
    end
end

"""
    check_point(M::EqualityManifold, p; kwargs...)

Checks whether ``h(x)=0`` where ``h`` is the defining function for ``M``. A tolerance can be given as part of the `kwargs`.
"""
function check_point(M::EqualityManifold, p; kwargs...)
    s = check_size(M, p)
    if !isnothing(s)
        return s
    end
    h = eval_defining_function(M, p)
    if !all(isapprox.(h, 0.0; kwargs...))
        return DomainError(
            h,
            "The point $(p) does not lie on the $(M) since the defining function has value $(h)."
        )
    end
    return nothing
end

"""
    check_vector(M::EqualityManifold, p, X; kwargs...)

Checks whether ``\\nabla h(p)^\\top X = 0``.
"""
function check_vector(M::EqualityManifold, p, X; kwargs...)
    s = check_point(M, p)
    if !isnothing(s)
        return s
    end
    ∇hp = eval_defining_jacobian(M, p)
    ∇hpX = ∇hp * X
    if !all(isapprox.(∇hpX, 0.0; kwargs...))
        println("!! ", ∇hp, X, ∇hpX)
        return DomainError(
            ∇hp * X,
            "The vector $(X) is not tangent to $(M) at $(p) since its product with the Jacobian has value $(∇hpX)."
        )
    end
    return nothing
end

####################################################################
# Tangent spaces bases computation.
####################################################################

default_basis(::EqualityManifold) = DefaultOrthonormalBasis()

"""
    get_basis(M::EqualityManifold, p, ::DefaultOrthonormalBasis)

Uses the defining function ``h`` for `M` and conputes a basis of ``T_p\\mathcal{M}`` as an orthonormal basis of ``\\ker(\\nabla h(x)^\\top)``.

# Warning

This implementation does not match the format intended in `ManifoldsBase`: it returns a `Matrix` whose columns are a basis of ``T_p\\mathcal{M}``.
"""
get_basis(::EqualityManifold, p, ::DefaultOrthonormalBasis)

function get_basis_orthonormal(M::EqualityManifold, p, N::AbstractNumbers; kwargs...)
    dim = manifold_dimension(M)
    B = DefaultOrthogonalBasis(N)
    ∇hp = eval_defining_jacobian(M, p)
    basis = nullspace(∇hp)
    r = rank(basis)
    r ≠ dim && error("Jacobian of the defining function for $(M) with dimension $(dim) has rank $(r) at $(p).")
    return basis
end

"""
    get_vector(::EqualityManifold, p, X, ::DefaultOrthonormalBasis)

Based on computing a basis of the tangent space with an SVD of the Jacobian of h.
"""
get_vector(::EqualityManifold, p, c, ::DefaultOrthonormalBasis)

function get_vector_orthonormal!(M::EqualityManifold, Y, p, c, N::AbstractNumbers)
    basis = get_basis(M, p, DefaultOrthonormalBasis(N))
    Y = basis * c
    return Y
end

"""
    get_coordinates(M::EqualityManifold, p, X, B::DefaultOrthonormalBasis)

A VERY TEMPORARY implementation that would allow, by a naive linear system resolution, to retrieve coefficients in a tangent space, from an embedded tangent vector. That is:
* p ∈ M
* X ∈ TxM ⊂ ℝ^n
* The result is c ∈ ℝ^dim(M) such that X = Bc where B has its columns being an orthonormal basis of TxM.
"""
get_coordinates(::EqualityManifold, p, X, ::DefaultOrthonormalBasis)

function get_coordinates_orthonormal(M::EqualityManifold, p, X, N::AbstractNumbers)
    B = get_basis(M, p, DefaultOrthonormalBasis(N))
    c = B \ X
    return c
end

####################################################################
# Projection
####################################################################

"""
    project(M::EqualityManifold, p)

Computes the metric projection of `p` on `M`.
"""
function ManifoldsBase.project(M::EqualityManifold, p)
    n = representation_size(M)[1]
    h(y) = eval_defining_function(M, y)
    m = length(h(p))

    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, y[1:n])
    @NLobjective(model, Min, 0.5 * sum((y[i] - p[i])^2 for i in 1:n))
    @NLconstraint(model, [j = 1:m], h(y)[j] == 0)

    optimize!(model)
    q = value.(y)
    return q
end

####################################################################
# Retractions
####################################################################

default_retraction_method(::EqualityManifold) = ProjectionRetraction()

"""
    retract(::EqualityManifold, p, X, ::ProjectionRetraction)
"""
retract(M::EqualityManifold, p, X, ::ProjectionRetraction)

function retract_project!(M::EqualityManifold, q, p, X)
    if !is_vector(M, p, X; atol = 1.0e-6)
        # error("Vector $(X) is not a tangent vector to $(M) at $(p). It can not be retracted.")
    end
    pX = p .+ X
    q = project(M, pX)
    return q
end

####################################################################
# Injectivity radii
# TODO. Change this hierarchy of definitions so that the injectivity_radius is called whenever the exponential map is defined, instead of relying on the Sphere only.
####################################################################

# Types of artificial lower bounds
abstract type AbstractInvertibilityBound end

"""
    ExactInvertibility

    When it exists, computes the exact value of the invertibility radius; i.e., the injectivity_radius of the exponential map in most cases.
"""
mutable struct ExactInvertibility <: AbstractInvertibilityBound end

"""
    OneOverSpectral <: AbstractInvertibilityBound
"""
mutable struct OneOverSpectral <: AbstractInvertibilityBound end

"""
    Computes a lower bound to the invertibility radius as

    \\frac{n}{\\max\\{\\lambda(\nabla^2 h_i(x)) : i\\in\\{1,...,p\\}\\}}.
"""
mutable struct NOverSpectral <: AbstractInvertibilityBound end

"""
    Computes a lower bound to the invertibility radius as

    \\frac{1}{\\sqrt{\\max\\{\\lambda(\nabla^2 h_i(x)) : i\\in\\{1,...,p\\}\\}}}.
"""
mutable struct OneOverSqrtSpectral <: AbstractInvertibilityBound end

"""
    Computes a lower bound to the invertibility radius as

    \\frac{n}{\\sqrt{\\max\\{\\lambda(\nabla^2 h_i(x)) : i\\in\\{1,...,p\\}\\}}}.
"""
mutable struct NOverSqrtSpectral <: AbstractInvertibilityBound end

"""
    invertibility_radius(M, p; m, b)

    TODO.
    Write the doc that describes this as a lower bound on the radius where the retraction is supposed to be invertible.
"""
function invertibility_radius(M::AbstractManifold, p; m::AbstractRetractionMethod, b::AbstractInvertibilityBound) end

invertibility_radius(M::Manifolds.Sphere, p, m::StabilizedRetraction, b::ExactInvertibility) = injectivity_radius(M, p, m)

"""
    TODO. Document here.
    This function precisely returns an artificial lower bound.
"""
invertibility_radius(M::EqualityManifold, p; m::AbstractRetractionMethod = default_retraction_method(M), b::AbstractInvertibilityBound = OneOverSpectral()) = invertibility_radius(M, p, m, b)

"""
    invertibility_radius(M::EqualityManifold, p, m::ProjectionRetraction, b::OneOverSpectral)

Return a lower bound on the [`invertibility_radius`](@ref) of the `ProjectionRetraction` as

```math
    \\frac{1}{\\max\\{\\lambda(H_{h_i}(x)) : i\\in\\{1,...,p\\}\\}}.
```

where ``H_{h_i}`` is the Hessian matrix of the defining subfunction ``h_i`` for `M`.
"""
function invertibility_radius(M::EqualityManifold, p, m::ProjectionRetraction, b::OneOverSpectral)
    hessians = eval_defining_hessians(M, p)
    spectral_radii = [maximum(abs, eigvals(hessian)) for hessian in hessians]
    return 1 / maximum(spectral_radii)
end

function invertibility_radius(M::EqualityManifold, p, m::ProjectionRetraction, b::NOverSpectral)
    n = representation_size(M)[1]
    hessians = eval_defining_hessians(M, p)
    spectral_radii = [maximum(abs, eigvals(hessian)) for hessian in hessians]
    return n / maximum(spectral_radii)
end

function invertibility_radius(M::EqualityManifold, p, m::ProjectionRetraction, b::OneOverSqrtSpectral)
    hessians = eval_defining_hessians(M, p)
    spectral_radii = [maximum(abs, eigvals(hessian)) for hessian in hessians]
    return 1 / sqrt(maximum(spectral_radii))
end

function invertibility_radius(M::EqualityManifold, p, m::ProjectionRetraction, b::NOverSqrtSpectral)
    n = representation_size(M)[1]
    hessians = eval_defining_hessians(M, p)
    spectral_radii = [maximum(abs, eigvals(hessian)) for hessian in hessians]
    return n / sqrt(maximum(spectral_radii))
end

function default_invertibility_bound(M::AbstractManifold, m::AbstractRetractionMethod) end

default_invertibility_bound(::EqualityManifold, ::ProjectionRetraction) = NOverSqrtSpectral()
default_invertibility_bound(::Manifolds.Sphere, ::StabilizedRetraction) = ExactInvertibility()

####################################################################
# Random choice of points
####################################################################

function ManifoldsBase.rand(M::EqualityManifold)
    n = representation_size(M)[1]
    p = Base.rand(Float64, n)
    projp = project(M, p)
    return projp
end

export EqualityManifold


export invertibility_radius, AbstractInvertibilityBound, OneOverSpectral, NOverSpectral, OneOverSqrtSpectral, NOverSqrtSpectral

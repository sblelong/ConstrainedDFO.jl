"""
This file contains the description of a submanifold of ℝ^n defined by a unique defining function.
This structure is used to represent the feasible set for equality-constrained problems when such set is a Riemannian submanifold of ℝ^n.
"""

using ManifoldsBase
import ManifoldsBase: representation_size, manifold_dimension, check_point, check_vector, retract_project!, get_embedding, default_retraction_method, check_size, default_basis, get_basis, get_basis_orthonormal, get_vector_orthonormal!, get_coordinates_orthonormal
using Manifolds

using LinearAlgebra
using JuMP
using Ipopt
using ForwardDiff

export EqualityManifold

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
end

####################################################################
# Some basic features/getters/setters of an `EqualityManifold`.
####################################################################

manifold_dimension(M::EqualityManifold) = M.dimension

representation_size(M::EqualityManifold) = (manifold_dimension(M) + 1,)

function get_embedding(M::EqualityManifold)
    return Euclidean(representation_size(M)...)
end

eval_defining_function(M::EqualityManifold, p) = M.defining_function(p)

function eval_defining_jacobian(M::EqualityManifold, p)
    h(x) = eval_defining_function(M, x)
    ∇hp = ForwardDiff.jacobian(h, p)
    return ∇hp
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
# Retractions
####################################################################

default_retraction_method(::EqualityManifold) = ProjectionRetraction()

"""
    retract(::EqualityManifold, p, X, ::ProjectionRetraction)
"""
retract(M::EqualityManifold, p, X, ::ProjectionRetraction)

function retract_project!(M::EqualityManifold, q, p, X)
    if !is_vector(M, p, X)
        error("Vector $(X) is not a tangent vector to $(M) at $(p). It can not be retracted.")
    end
    n = representation_size(M)[1]
    h(y) = eval_defining_function(M, y)
    m = length(h(p))
    pX = p .+ X

    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, y[1:n])
    @NLobjective(model, Min, 0.5 * sum((y[i] - pX[i])^2 for i in 1:n))
    @NLconstraint(model, [j = 1:m], h(y)[j] == 0)

    optimize!(model)
    q = value.(y)
    return q
end

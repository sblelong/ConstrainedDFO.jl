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

export EqualityManifold, eval_defining_function

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

representation_size(M::EqualityManifold) = (M.dimension + 1,)

manifold_dimension(M::EqualityManifold) = M.dimension

eval_defining_function(M::EqualityManifold, p) = M.defining_function(p)

function eval_defining_jacobian(M::EqualityManifold, p)
    h(x) = eval_defining_function(M, x)
    ∇hp = ForwardDiff.jacobian(h, p)
    return ∇hp
end

function check_point(M::EqualityManifold, p; kwargs...)
    h = eval_defining_function(M, p)
    if !all(isapprox.(h, 0.0; kwargs...))
        return DomainError(
            h,
            "The point $(p) does not lie on the $(M) since the defining function has value $(h)."
        )
    end
    return nothing
end

function check_vector(M::EqualityManifold, p, X; kwargs...)
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

"""
    get_vector(::EqualityManifold, p, X, ::DefaultOrthonormalBasis)

Based on computing a basis of the tangent space with an SVD of the Jacobian of h.
"""
get_vector(::EqualityManifold, p, X, ::DefaultOrthonormalBasis)

function get_vector_orthonormal!(M::EqualityManifold, Y, p, c, N::AbstractNumbers)
    basis = get_basis(M, p, DefaultOrthonormalBasis())
    println(c)
    Y = basis * c
    return Y
end

"""
    retract(::EqualityManifold, p, X, ::ProjectionRetraction)
"""
retract(M::EqualityManifold, p, X, ::ProjectionRetraction)

function retract_project!(M::EqualityManifold, q, p, X)
    # check_vector(M, p, X; error = :warn)
    n = representation_size(M)[1]
    h(y) = eval_defining_function(M, y)
    m = length(h(p))
    pX = p + get_vector(M, p, X, DefaultOrthonormalBasis())

    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, y[1:n])
    @NLobjective(model, Min, 0.5 * sum((y[i] - pX[i])^2 for i in 1:n))
    @NLconstraint(model, [j = 1:m], h(y)[j] == 0)

    optimize!(model)
    q = value.(y)
    return q
end

function get_embedding(M::EqualityManifold)
    return Euclidean(representation_size(M)...)
end

default_retraction_method(::EqualityManifold) = ProjectionRetraction()

function check_size(M::EqualityManifold, p)
    if length(p) ≠ representation_size(M)[1]
        return DomainError("Vector $(p) cannot belong to $(M) with representation size $(representation_size(M)): it has length $(length(p)).")
    else
        return nothing
    end
end
function check_size(M::EqualityManifold, p, X)
    if length(X) ≠ representation_size(M)[1]
        return DomainError("Vector $(X) cannot be a tangent vector to $(M) with dimension $(manifold_dimension(M)): it has length $(length(X)).")
    else
        return check_size(M, p)
    end
end

default_basis(::EqualityManifold) = DefaultOrthonormalBasis()

"""
    get_basis(M::EqualityManifold, p, ::DefaultOrthonormalBasis)

Uses the defining function ``h`` for `M` and conputes a basis of ``T_p\\mathcal{M}`` as an orthonormal basis of ``\\ker(\\nabla h(x)^\\top)``.
"""
get_basis(::EqualityManifold, p, ::DefaultOrthonormalBasis)

function get_basis_orthonormal(M::EqualityManifold, p, N::AbstractNumbers; kwargs...)
    dim = manifold_dimension(M)
    B = DefaultOrthogonalBasis(N)
    ∇hp = eval_defining_jacobian(M, p)
    basis = nullspace(∇hp)
    r = rank(basis)
    println("Returning basis with size $(r)")
    r ≠ dim && error("Jacobian of the defining function for $(M) with dimension $(dim) has rank $(r) at $(p).")
    return basis
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

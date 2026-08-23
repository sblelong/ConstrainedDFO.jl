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

manifold_dimension(M::EqualityManifold) = M.dimension

representation_size(M::EqualityManifold) = (M.embedding_dimension,)

function get_embedding(M::EqualityManifold)
    return Euclidean(representation_size(M)...)
end

eval_defining_function(M::EqualityManifold, p) = M.defining_function(p)

function eval_defining_jacobian(M::EqualityManifold, p)
    h(x) = eval_defining_function(M, x)
    Jhp = jacobian(h, p)
    return Jhp
end

function eval_defining_hessian(M::EqualityManifold, p, i::Int)
    hi(x) = eval_defining_function(M, x)[i]
    Hhip = hessian(hi, p)
    return Hhip
end

"""
    eval_defining_hessians(M::EqualityManifold, p)
"""
function eval_defining_hessians(M::EqualityManifold, p)
    nb_defining_functions = length(eval_defining_function(M, p))
    Hhis = Matrix[]
    for i in 1:nb_defining_functions
        push!(Hhis, eval_defining_hessian(M, p, i))
    end
    return Hhis
end

function check_size(M::EqualityManifold, p)
    if size(p) ≠ representation_size(M)
        return DomainError("Vector $(p) cannot belong to $(M) with representation size $(representation_size(M)): it has length $(length(p)).")
    else
        return nothing
    end
end

function check_size(M::EqualityManifold, p, X)
    if size(X) ≠ representation_size(M)
        return DomainError("Vector $(X) cannot be a tangent vector to $(M) with dimension $(manifold_dimension(M)): it has length $(length(X)).")
    else
        return check_size(M, p)
    end
end

function check_point(M::EqualityManifold, p; kwargs...)
    s = check_size(M, p)
    if !isnothing(s)
        return s
    end
    h = eval_defining_function(M, p)
    if !all(isapprox.(h, 0.0; kwargs...))
        return DomainError(
            h,
            "The point $(p) does not lie on the $(M): h(p)=$(h)."
        )
    end
    return nothing
end

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

default_basis(::EqualityManifold) = DefaultOrthonormalBasis()

get_basis(::EqualityManifold, p, ::DefaultOrthonormalBasis)

function get_basis_orthonormal(M::EqualityManifold, p, N::AbstractNumbers; kwargs...)
    dim = manifold_dimension(M)
    B = DefaultOrthogonalBasis(N)
    Jhp = eval_defining_jacobian(M, p)
    basis = nullspace(Jhp)
    r = rank(basis)
    r ≠ dim && error("Jacobian of the defining function for $(M) with dimension $(dim) has rank $(r) at $(p).")
    return basis
end

get_vector(::EqualityManifold, p, c, ::DefaultOrthonormalBasis)

function get_vector_orthonormal!(M::EqualityManifold, Y, p, c, N::AbstractNumbers)
    basis = get_basis(M, p, DefaultOrthonormalBasis(N))
    Y = basis * c
    return Y
end

get_coordinates(::EqualityManifold, p, X, ::DefaultOrthonormalBasis)

function get_coordinates_orthonormal(M::EqualityManifold, p, X, N::AbstractNumbers)
    B = get_basis(M, p, DefaultOrthonormalBasis(N))
    c = B \ X
    return c
end

function project(M::EqualityManifold, p)
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

default_retraction_method(::EqualityManifold) = ProjectionRetraction()

function retract_project!(M::EqualityManifold, q, p, X)
    if !is_vector(M, p, X; atol = 1.0e-6)
        error("Vector $(X) is not a tangent vector to $(M) at $(p). It can not be retracted.")
    end
    pX = p .+ X
    q = project(M, pX)
    return q
end

function ManifoldsBase.rand(M::EqualityManifold)
    n = representation_size(M)[1]
    p = Base.rand(Float64, n)
    projp = project(M, p)
    return projp
end

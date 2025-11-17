"""
This file contains the description of a submanifold of ℝ^n defined by a unique defining function.
It serves as the feasible set for equality constraints for problems with such constraints.
"""

using ManifoldsBase
import ManifoldsBase: representation_size, manifold_dimension, check_point

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

function check_point(M::EqualityManifold, p; kwargs...)
    h = eval_defining_function(M, p)
    if !isapprox(h, 0.0; kwargs...)
        return DomainError(
            h,
            "The point $(p) does not lie on the $(M) since the defining function has value $(h)."
        )
    end
    return nothing
end

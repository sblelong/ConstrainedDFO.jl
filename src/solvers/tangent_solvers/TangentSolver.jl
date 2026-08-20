"""
    AbstractTangentSolver

An abstract type for all solvers used to solve subproblems in tangent spaces with the [`DFROSolver`](@ref).
"""
abstract type AbstractTangentSolver end

"""
    solve!(TS::AbstractTangentSolver, M::AbstractManifold, p, R::AbstractRetractionMethod, invertibility_radius; g)

Solve the subproblem

```math
    \\begin{array}{{c r @{\\;} c @{\\;} l}}
        \\min\\limits_{v\\in T_p\\mathcal{M}} & f\\circ R_p(v)            \\
        \\mathrm{s.t.}               & g\\circ R_p(v) & \\leq & 0
    \end{array}
```

with the tangent solver `TS`. Stops whenever an iterate (i.e., a feasible improving point) is found outside of the [`invertibility_radius`](@ref). A history of all tangent iterates, associated retractions and (`f`,`g`) values is stored within the `AbstractTangentSolver` object.
"""
function solve!(TS::AbstractTangentSolver, M::AbstractManifold, p, invertibility_radius; g) end

"""
    get_last_subproblem_result(TS::AbstractTangentSolver)

Retrieve all data stored within the `AbstractTangentSolver` object. Contains:
- values of all evaluated points within ``\\mathbb{R}^q``;
- associated retractions on the manifold the solver was last called on;
- associated values of the objective function and inequality constraints.
"""
function get_last_subproblem_result(TS::AbstractTangentSolver) end

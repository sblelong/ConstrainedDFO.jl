"""
    AbstractTangentSolver

An abstract type for all solvers used to solve subproblems in tangent spaces with the [`DFROSolver`](@ref).
"""
abstract type AbstractTangentSolver end

"""
    BlackboxDataType

A structure representing all the data expected to be returned by an [`AbstractTangentSolver`](@ref) evaluating a blackbox.
"""
mutable struct BlackboxDataType
    x::Vector{Float64}
    f::Float64
    g::Vector{Float64}
end

function _store_eval_data!(TS::AbstractTangentSolver, eval_data::BlackboxDataType)
    push!(TS.data_Rpv, eval_data.x)
    push!(TS.data_f, eval_data.f)
    length(eval_data.g) > 0 && push!(TS.data_g, eval_data.g)
    return TS
end

function retract_eval_store!(
        TS::AbstractTangentSolver,
        M::AbstractManifold,
        p,
        R::AbstractRetractionMethod,
        mco::AbstractManifoldCostObjective,
        n_ineqs::Int,
        g,
        v;
        εeqs::Float64 = 1.0e-8
    )
    d = get_vector(M, p, v, DefaultOrthonormalBasis())
    Rpv = retract(M, p, d, R)

    fRpv = is_point(M, Rpv; atol = εeqs) ? get_cost(M, mco, Rpv) : FAILURE_MAX

    if n_ineqs > 0
        gRpv = g(Rpv)
    else
        gRpv = Float64[]
    end

    eval_data = BlackboxDataType(Rpv, fRpv, gRpv)

    _store_eval_data!(TS, eval_data)

    return eval_data
end

"""
    format_eval_data(TS::AbstractTangentSolver, eval_data::BlackboxDataType)

This should be implemented for all concrete types inheriting from [`AbstractTangentSolver`](@ref) in order to convert the data stored inside a [`BlackboxDataType`](@ref) object in a format readble by the tangent solver.
"""
function format_eval_data(TS::AbstractTangentSolver, eval_data::BlackboxDataType) end

"""
    blackbox_wrapper_store!(TS::AbstractTangentSolver, M::AbstractManifold, p, R::AbstractRetractionMethod, f, n_ineqs::Int, g, v)

Retract the tangent vector `v` to the manifold `M` and evaluate the blackbox made of the objective function `f` and the inequality constraints `g`. The results are stored within the relevant attributes of the [`AbstractTangentSolver`](@ref).
"""
function blackbox_wrapper_store!(
        TS::AbstractTangentSolver,
        M::AbstractManifold,
        p,
        R::AbstractRetractionMethod,
        mco::AbstractManifoldCostObjective,
        n_ineqs::Int,
        g,
        v;
        εeqs::Float64 = 1.0e-8
    )
    eval_data = retract_eval_store!(TS, M, p, R, mco, n_ineqs, g, v; εeqs)
    formatted_data = format_eval_data(TS, eval_data)
    return formatted_data
end

"""
    solve!(TS::AbstractTangentSolver, f, M::AbstractManifold, p, R::AbstractRetractionMethod, ρ::AbstractInvertibilityBound; g)

Solve the subproblem

```math
    \\begin{array}{{c r @{\\;} c @{\\;} l}}
        \\min\\limits_{v\\in T_p\\mathcal{M}} & f\\circ R_p(v)            \\
        \\mathrm{s.t.}                        & g\\circ R_p(v) & \\leq & 0
    \\end{array}
```

with the tangent solver `TS`. Stops whenever an iterate (i.e., a feasible improving point) is found outside of the [`invertibility_radius`](@ref), with bound given by `ρ` (see [`AbstractInvertibilityBound`](@ref)).

A history of all tangent iterates, associated retractions and (`f`,`g`) values is stored within the `AbstractTangentSolver` object.
"""
function solve!(
        TS::AbstractTangentSolver,
        f,
        M::AbstractManifold,
        p,
        R::AbstractRetractionMethod,
        ρ::AbstractInvertibilityBound,
        n_ineqs::Int;
        g, max_evals::Int, εeqs::Float64 = 1.0e-8
    )
end

"""
    get_last_subproblem_result(TS::AbstractTangentSolver)

Retrieve all data stored within the `AbstractTangentSolver` object. Contains:
- values of all evaluated points within ``\\mathbb{R}^q``;
- associated retractions on the manifold the solver was last called on;
- associated values of the objective function and inequality constraints.
"""
function get_last_subproblem_result(TS::AbstractTangentSolver) end

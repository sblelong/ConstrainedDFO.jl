abstract type AbstractBenchmarkProblem end

mutable struct BenchmarkEqualityProblem <: AbstractBenchmarkProblem
    dimension::Int
    objective::Function
    ineq_constraints::Function
    eq_constraints::Function
    x0::Vector{Float64}
    A::Union{Nothing, Matrix{Float64}}
    b::Union{Nothing, Vector{Float64}}
end

function BenchmarkEqualityProblem(dimension::Int, objective::Function, ineq_constraints::Function, eq_constraints::Function, x0; A::Union{Nothing, Matrix{Float64}} = nothing, b::Union{Nothing, Vector{Float64}} = nothing)
    return BenchmarkEqualityProblem(dimension, objective, ineq_constraints, eq_constraints, x0, A, b)
end

mutable struct BenchmarkManifoldProblem <: AbstractBenchmarkProblem
    manifold::AbstractManifold
    objective::Function
    ineq_constraints::Function
    x0::Vector{Float64}
end

get_dimension(BPE::BenchmarkEqualityProblem) = BPE.dimension
get_dimension(BPM::BenchmarkManifoldProblem) = length(BPM.x0)

eval_obj(BP::AbstractBenchmarkProblem, x) = BP.objective(x)

eval_ineqs(BP::AbstractBenchmarkProblem, x) = BP.ineq_constraints(x)

eval_eqs(BPE::BenchmarkEqualityProblem, x) = BPE.eq_constraints(x)

get_x0(BP::AbstractBenchmarkProblem) = BP.x0

nb_inequality_constraints(BP::AbstractBenchmarkProblem) = length(eval_ineqs(BP, get_x0(BP)))
has_inequality_constraints(BP::AbstractBenchmarkProblem) = nb_inequality_constraints(BP) > 0

function get_equality_manifold(BPE::BenchmarkEqualityProblem)
    x0 = get_x0(BPE)
    n = length(x0)
    h(x) = eval_eqs(BPE, x)
    h0 = h(x0)
    p = length(h0)
    M = EqualityManifold(h, n - p, n)
    return M
end
get_equality_manifold(BPM::BenchmarkManifoldProblem) = BPM.manifold

export BenchmarkEqualityProblem, BenchmarkManifoldProblem, get_dimension, eval_obj, eval_ineqs, eval_eqs, get_x0, get_equality_manifold, has_inequality_constraints, nb_inequality_constraints

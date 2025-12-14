export AbstractEvalManager, FractionEvalManager, get_eval_budget, get_remaining_evals, update_remaining_evals!

abstract type AbstractEvalManager end

mutable struct DefaultEvalManager <: AbstractEvalManager
    used_evals::Int
    remaining_evals::Int
end
DefaultEvalManager(max_evals::Int) = DefaultEvalManager(0, max_evals)

"""
    get_eval_budget(em::AbstractEvalManager)
"""
get_eval_budget(dem::DefaultEvalManager) = dem.remaining_evals

function get_remaining_evals(dem::DefaultEvalManager)
    return dem.remaining_evals
end

"""
    update_remaining_evals(dem::DefaultEvalManager)
"""
function update_remaining_evals!(dem::DefaultEvalManager, used_evals::Int)
    dem.used_evals += used_evals
    return dem.remaining_evals -= used_evals
end

mutable struct FractionEvalManager <: AbstractEvalManager
    base_manager::DefaultEvalManager
    fraction::Float64
end
FractionEvalManager(max_evals::Int, fraction::Float64) = FractionEvalManager(DefaultEvalManager(max_evals), fraction)

function get_eval_budget(fem::FractionEvalManager)
    return Int(floor(get_remaining_evals(fem.base_manager) * fem.fraction))
end

update_remaining_evals!(fem::FractionEvalManager, used_evals::Int) = update_remaining_evals!(fem.base_manager, used_evals)

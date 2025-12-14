module ConstrainedDFO

include("structures/EqualityManifold.jl")
include("structures/StoppingCriteria.jl")
include("structures/RDFOState.jl")
include("structures/EvalManager.jl")

include("solvers/Solvers.jl")
# using .Solvers

include("benchmark/Benchmark.jl")
# using .Benchmark

include("tools/Tools.jl")
# using .Tools

end

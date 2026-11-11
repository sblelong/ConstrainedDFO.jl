# How to run benchmarks with ConstrainedDFO.jl?

This directory contains scripts to run 4 benchmark procedures. Each procedure solves some instances of the [CUTEst](https://github.com/ralna/CUTEst) test suite and formats the output to make plotting easier. The purpose of each of these benchmark procedures is detailed in the paper introducing the DFRO method.

## Prerequisites

- All Julia dependencies are gathered in the `benchmark/Project.toml` file. From the source of this repo, run:
```julia
julia> using Pkg; Pkg.activate("benchmark/")
```
- As is standard in the field of DFO, performance of algorithms is assessed through data and performance profiles [^1][^2]. The postprocessing scripts here are meant to work with the [RunnerPost](https://github.com/bbopt/RunnerPost) software.

## Choice of instances

Each benchmark is run on a specific set of CUTEst instances. Selection of each testbed is made in the `utils/problem_selection.jl` file.

## Running benchmarks

Each benchmarking script:
- selects relevant instances;
- solves them with the solvers to be compared and stores logs in the `benchmark/logs/<benchmark_name>/<solver_name>` directory;
- performs postprocessing of the log files in `benchmarks/runnerpost/<benchmark_name>/<solver_name>` to match the format expected by the RunnerPost software;
- creates the `.def` files needed for profiles to be plotted with the RunnerPost software.

The user is only left with calling the RunnerPost software on the postprocessed data to reproduce figures presented in the original paper. From the `benchmark` directory, the pipeline looks as follows:
```
julia --project=. <benchmark_name>.jl
cd runnerpost/<benchmark_name>
/path/to/runnerpost/build/release/bin/runnerPost algo_selection.def problem_selection.def output_selection.def
```

*Note:* These instructions are not updated and may not work with future versions of the RunnerPost software. Users are referred to the software's documentation.

[^1]: J.J. Moré and S.M. Wild. Benchmarking Derivative-Free Optimization Algorithms. *SIAM Journal on Optimization*, 20(1):172–191, 2009. [doi:10.1137/080724083](https://doi.org/10.1137/080724083)
[^2]: E.D. Dolan and J.J. Moré. Benchmarking optimization software with performance profiles. *Mathematical Programming*, 91(2):201–213, 2002. [doi:10.1007/s101070100263](https://doi.org/10.1007/s101070100263)
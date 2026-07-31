using Documenter, ConstrainedDFO

makedocs(
    sitename="ConstrainedDFO.jl",
    remotes = nothing,
    modules = [ConstrainedDFO]
)

deploydocs(
    repo = "github.com/sblelong/ConstrainedDFO.jl.git",
)
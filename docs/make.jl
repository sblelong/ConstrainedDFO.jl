using ConstrainedDFO
using Documenter
using DocumenterInterLinks

links = InterLinks(
    "ManifoldsBase" => ("https://juliamanifolds.github.io/ManifoldsBase.jl/stable/"),
    "Manopt" => ("https://manoptjl.org/stable/")
)

makedocs(
    sitename = "ConstrainedDFO.jl",
    plugins = [links]
)

deploydocs(
    repo = "github.com/sblelong/ConstrainedDFO.jl.git",
)

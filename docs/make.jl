using Documenter
using StateSpaceLearning

# Set up to run docstrings with jldoctest
DocMeta.setdocmeta!(
    StateSpaceLearning, :DocTestSetup, :(using StateSpaceLearning); recursive=true
)

makedocs(;
    modules=[StateSpaceLearning],
    doctest=true,
    clean=true,
    format=Documenter.HTML(mathengine=Documenter.MathJax2()),
    sitename="StateSpaceLearning.jl",
    authors="André Ramos",
    pages=[
        "Home" => "index.md",
        "manual.md", "adapting_package.md"
    ],
)

deploydocs(
        repo="github.com/LAMPSPUC/StateSpaceLearning.jl.git",
        push_preview = true
    )

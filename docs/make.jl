using Documenter, DocumenterVitepress
using EasyHybrid

makedocs(;
    modules=[EasyHybrid],
    authors="Lazaro Alonso, Markus Reichstein, Bernhard Ahrens",
    repo="https://github.com/EarthyScience/EasyHybrid",
    sitename="EasyHybrid",
    format=DocumenterVitepress.MarkdownVitepress(
        repo = "https://github.com/EarthyScience/EasyHybrid",
        devurl = "dev",
    ),
    pages=[
        "Home" => "index.md",
        "Research" =>[
            "RbQ10" => "research/RbQ10_results.md"
            "BulkDensitySOC" => "research/BulkDensitySOC_results.md"
        ],
        "API" => "api.md",
    ],
    warnonly = true,
)

deploydocs(;
    repo="github.com/EarthyScience/EasyHybrid",
    push_preview=true,
)

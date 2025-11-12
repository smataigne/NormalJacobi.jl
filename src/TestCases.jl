using LinearAlgebra, SkewLinearAlgebra, Plots, LaTeXStrings, BenchmarkTools
include("NormalJacobi.jl")
include("NormalJacobi2.jl")
include("Utils.jl")

A = create_matrix(200, 0., 0.0) #(20,0.3,0.3)
display(norm(A*A'-A'A)/norm(A)^2)
display(eigvals(A))

"""
P = plot(framestyle=:box, legend=:topright,font="Computer Modern", tickfontfamily="Computer Modern",legendfont="Computer Modern", guidefontfamily = "Computer Modern",
legendfontsize=10,yguidefontsize=13,xguidefontsize=13, xtickfontsize = 13, ytickfontsize=13, margin = 0.3Plots.cm, minorgrid = false, titlefontfamily="Computer Modern", titlefontsize=13)
heatmap!(log10.(max.(abs.(A), eps(Float64))), colormap=:viridis, xticks=false, yticks=false, colorbar_fontsize=1, clim=(-15, 0); yflip=true)
#savefig(P, "NormalJacobi_phaseX.pdf")
display(P)
"""

@btime normalskewjacobi(A)
@btime normaljacobi(A)
#T = normalskewjacobi!(A, false)
#display(T)










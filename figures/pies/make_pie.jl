using Plots, LaTeXStrings, LinearAlgebra, SkewLinearAlgebra
include("NormalJacobi.jl")
include("UtilsforTests.jl")
#=
This script runs the time experiments for the normal Jacobi algorithm and generates pie charts to visualize the time distribution across different steps of the algorithm for various test cases. The results are saved as a PDF file.
timesE1 = zeros(3)
timesE2 = zeros(3)
timesE3 = zeros(3)
timesE4 = zeros(3)
timesE5 = zeros(3)

N = 10
n = 400

for _ ∈ 1:N
     #Case E1 Haar-distributed random orthogonal matrix
        A = Matrix(qr(randn(n, n)).Q)
        _, t = time_normal_jacobi_skew!(A)
        timesE1 .+= t ./ N
        print("Case 1: done \n")
        #Case E2 Random normal matrix
        A = create_matrix_rand(n, 0.0 , 0.0)
        _, t = time_normal_jacobi_skew!(A)
        timesE2 .+= t ./ N
        print("Case 2: done \n")
        #Case E3 Haar-distributed Schur vectors, random eigenvalues with 30% real eigenvalues
        A = create_matrix(n, 0.34, 0.0)
        _, t = time_normal_jacobi_skew!(A)
        timesE3 .+= t ./ N
        print("Case 3: done \n")
        #Case E4 Haar-distributed Schur vectors, random eigenvalues with 30% repeated complex eigenvalues
        A = create_matrix(n, 0.0, 0.34)
        _, t = time_normal_jacobi_skew!(A)
        timesE4 .+= t ./ N
        print("Case 4: done \n")
        #Case E5 Haar-distributed Schur vectors, worst random eigenvalues
        A = create_matrix_worst(n, 0.0, 0.0)
        _, t = time_normal_jacobi_skew!(A)
        timesE5 .+= t ./ N
        print("Case 5: done \n")  
end
=#
labels = ["Step I"; "Step II"; "Step III"]
# Créer les pie charts sans légende
P1 = pie(labels, timesE1, legend = false, title = "Exp. 1", margin = 0Plots.mm, spacing = -15Plots.mm)
P2 = pie(labels, timesE2, legend = false, title = "Exp. 2", margin = 0Plots.mm, spacing = -15Plots.mm)
P3 = pie(labels, timesE3, legend = false, title = "Exp. 3", margin = 0Plots.mm, spacing = -15Plots.mm)
P4 = pie(labels, timesE4, legend = false, title = "Exp. 4", margin = 0Plots.mm, spacing = -15Plots.mm)
P5 = pie(labels, timesE5, legend = false, title = "Exp. 5", margin = 0Plots.mm, spacing = -15Plots.mm)

# Créer un plot fantôme (invisible) qui porte uniquement la légende
n = length(labels)
P_legend = plot(
    [NaN], [NaN],
    seriestype = :shape,
    label = labels[1],
    legend = :inside,
    legendcolumns = n,
    framestyle = :none,
    axis = false,
    grid = false,
    margin = 0Plots.mm
)
for i in 2:n
    plot!(P_legend, [NaN], [NaN], seriestype = :shape, label = labels[i])
end
# Assembler : 3 pie charts + 1 plot légende en bas
P = plot(
    P1, P2, P3, P4, P5, P_legend,
    layout = @layout([a b c d e; f{0.01h}]),   # 5% instead of 10% for legend row
    spacing = 0,
    size = (1500, 350),                     # reduce horizontal gaps between pies
    ticksfontfamily    = "Computer Modern",
    fontfamily         = "Computer Modern",
    legendfontfamily   = "Computer Modern",
    titlefontfamily    = "Computer Modern",
    legendfontsize  = 20,
    labelfontsize   = 20,
    tickfontsize    = 20,
    titlefontsize   = 20,
)

display(P)
savefig(P, joinpath(@__DIR__, "../figures/experiments_time_pie.pdf"))
println("done")

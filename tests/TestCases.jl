using LinearAlgebra, SkewLinearAlgebra, Plots, LaTeXStrings, BenchmarkTools
include("../src/NormalJacobi.jl")
include("../src/Utils.jl")

A = create_matrix2(20,0.3,0.3)
display(norm(A*A'-A'A)/norm(A)^2)
normal_skew_jacobi!(A, true)










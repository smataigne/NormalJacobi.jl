using LinearAlgebra, SkewLinearAlgebra,LaTeXStrings, BenchmarkTools
include("../src/NormalJacobiforPlots.jl")
include("./UtilsforTests.jl")
BLAS.set_num_threads(1)  # Ensure single-threaded execution for fair timing
n = 20
A = create_matrix2(n, 0.3, 0.3)
normal_skew_jacobi_plots(A)










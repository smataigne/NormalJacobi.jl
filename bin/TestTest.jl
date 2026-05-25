using LinearAlgebra, BenchmarkTools
include("UtilsforTests.jl")
include("../src/NormalJacobiHe.jl")
include("../src/NormalJacobiZhou.jl")
include("../src/NormalJacobiBunse.jl")
include("../src/NormalJacobi.jl")

#=
n = 4
A = create_matrix_worst(n, 0.0, 0.0)
B = copy(A)
normal_jacobi_bunse!(B)
offSchur(B) / norm(A)

M = randn(8, 8)
M = (M + M')/2
D = Diagonal(rand(4))
S  = [zeros(4, 4) -D; D zeros(4, 4)]
A = M + S
display(norm(A'A - A*A'))
=#
BLAS.set_num_threads(1)
for  _ in 1:10
    n = 128
    A = create_matrix(n, 0.0, 0.34)
    @time normal_jacobi_skew!(A)
    display(offschur(A))
end
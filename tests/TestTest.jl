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
n = 50
A = Matrix(qr(randn(n, n)).Q)
@btime normal_jacobi_bunse2!(copy(A), Array(1:n))
@btime normal_jacobi_bunse!(copy(A))
println("Done")
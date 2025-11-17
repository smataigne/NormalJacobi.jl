using LinearAlgebra
include("../src/NormalJacobi.jl")
include("../src/NormalJacobiBunse.jl")
include("../src/NormalJacobiGoldstine.jl")
include("../src/NormalJacobiZhou.jl")

ns = [32, 100,316]
N = length(ns)
NE = 3
errors = zeros(NE, N, 4)

@views function testmatrix!(A::AbstractMatrix, errors::AbstractVector)
    B = copy(A)
    normal_skew_jacobi!(B, false)
    errors[1] = norm(offSchur(B)) / norm(A)
    B = copy(A)
    normal_jacobi_bunse!(B)
    errors[2] = norm(offSchur(B)) / norm(A)
    B = complex.(copy(A), 0)
    normal_jacobi_goldstine!(B) 
    errors[3] = norm(offdiag(B)) / norm(A)
    B = copy(A)
    normal_jacobi_zhou!(B)
    errors[4] = norm(offSchur(B)) / norm(A)
    return errors
end

for (i, n) ∈ enumerate(ns)
    #Case E1
    A = Matrix(qr(randn(n, n)).Q)
    errors[1, i, :] = testmatrix!(A, errors[1, i, :])
    print("Case 1: done \n")
    #Case E2
    A = create_matrix(n, 0.3, 0.0)
    errors[2, i, :] = testmatrix!(A, errors[2, i, :])
    print("Case 2: done \n")
    #Case E3
    A = create_matrix(n, 0.0, 0.3)
    errors[3, i, :] = testmatrix!(A, errors[3, i, :])
    print("Case 3: done \n")
end


errors2 = round.(errors; sigdigits = 2)
for k ∈ 1:NE
    for (i, n) ∈ enumerate(ns)
        print(n ,"&", errors2[k, i, 1], "&", errors2[k, i, 2], "&", errors2[k, i, 3], "&", errors2[k, i, 4], "\\\\ \n")
    end
end
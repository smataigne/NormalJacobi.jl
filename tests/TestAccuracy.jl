using LinearAlgebra
include("../src/NormalJacobi.jl")
include("../src/NormalJacobiBunse.jl")
include("../src/NormalJacobiGoldstine.jl")
include("../src/NormalJacobiZhou.jl")
include("../src/NormalJacobiHe.jl")
include("./UtilsforTests.jl")

ns =[64, 128]#,256,512]
N = length(ns)
NE = 5  # Number of experiments
errors = zeros(NE, N, 5)
temp  = zeros(5)
T = 1 # Number of trials per case

@views function testmatrix!(A::AbstractMatrix, errors::AbstractVector)
    B = copy(A)
    normal_jacobi_skew!(B)
    errors[1] = offschur(B) / norm(A)
    B = copy(A)
    normal_jacobi_bunse!(B)
    errors[2] = offschur(B) / norm(A)
    B = complex.(copy(A), 0)
    normal_jacobi_goldstine!(B) 
    errors[3] = offdiag(B) / norm(A)
    B = copy(A)
    normal_jacobi_zhou!(B)
    errors[4] = offschur(B) / norm(A)
    B = complex.(copy(A), 0)
    normal_jacobi_he!(B)
    errors[5] =  offdiag(B) /  norm(A)
    return log.(errors)
end

for (i, n) ∈ enumerate(ns)
    println("Entering n = ", n)
    for t ∈ 1:T
        #Case E1 Haar-distributed random orthogonal matrix
        A = Matrix(qr(randn(n, n)).Q)
        @elapsed errors[1, i, :] .+= testmatrix!(A, temp)
        print("Case 1: done \n")
        #Case E2 Random normal matrix
        A = create_matrix_rand(n, 0.0 , 0.0)
        @elapsed errors[2, i, :] .+= testmatrix!(A, temp)
        print("Case 2: done \n")
        #Case E3 Haar-distributed Schur vectors, random eigenvalues with 30% real eigenvalues
        A = create_matrix(n, 0.34, 0.0)
        @elapsed errors[3, i, :] .+= testmatrix!(A, temp)
        print("Case 3: done \n")
        #Case E4 Haar-distributed Schur vectors, random eigenvalues with 30% repeated complex eigenvalues
        A = create_matrix(n, 0.0, 0.34)
        @elapsed errors[4, i, :] .+= testmatrix!(A, temp)
        print("Case 4: done \n")
        #Case E5 Haar-distributed Schur vectors, worst random eigenvalues
        A = create_matrix_worst(n, 0.0, 0.0)
        @elapsed errors[5, i, :] .+= testmatrix!(A, temp)
        print("Case 5: done \n")
    end
end

errors .= exp.(errors ./ T) #Geometric mean


errors2 = round.(errors; sigdigits = 2)
for k ∈ 1:NE
    for (i, n) ∈ enumerate(ns)
        print(n ,"&", errors2[k, i, 1], "&", errors2[k, i, 2], "&", errors2[k, i, 3], "&", errors2[k, i, 4], "&", errors2[k, i, 5], "\\\\ \n")
    end
end
using LinearAlgebra

n = 8
A = zeros(Int, n, n)
for i ∈ 1:2:n-2
    for j ∈ i+2:n
        A[i, j] = 1
        A[j, i] = 1
        A[i + 1, j] = 1
        A[j, i + 1] = 1
    end
end
println("The following matrix is 1 everywhere except for the diagonal 2x2 blocks, which are 0:")
display(A)
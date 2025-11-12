using LinearAlgebra

"""
See
Heike Faßbender, D.Steven Mackey, Niloufer Mackey, Hamilton and Jacobi come full circle: Jacobi algorithms for structured Hamiltonian eigenproblems,
Linear Algebra and its Applications, Volumes 332–334, 2001, Pages 37-80, ISSN 0024-3795,
https://doi.org/10.1016/S0024-3795(00)00093-8.
"""

function offdiag(A::AbstractMatrix)
    Σ = 0
    n = size(A, 1)
    for i ∈ 1:n
        for j ∈ (i+1):n
            Σ += A[i, j]^2
        end
    end 
    return sqrt(Σ)
end

"""
    SSHjacobi!(A::AbstractMatrix{T}) where T

Jacobi-like algorithm for symmetric skew-Hamiltonian matrices.
"""
@views function SSHjacobi!(A::AbstractMatrix{T}) where T
    n = size(A, 1)
    itermax = 10; iter = 1
    n2 = n ÷ 2
    ii = zeros(Int64, 4)
    p = zeros(T, 3)
    R = zeros(T, 4, 4)
    ε = eps(T) * 100 * norm(A)
    while offdiag(A) > ε && iter < itermax
        for i ∈ 1:n2-1
            for j ∈ i+1:n2
                ii .= i, j, i+n2, j+n2
                M = A[ii, ii]
                p .= -M[1, 4], 0.5*(M[1, 1] - M[2, 2]), M[1, 2]
                α = norm(p)
                β = α + p[2]
                R[1, :] .= β, p[3], 0, -p[1]
                R[2, :] .= -p[3], β, -p[1], 0
                R[3, :] .= 0, p[1], β, p[3]
                R[4, :] .= p[1], 0, -p[3], β
                R .*= (1 / √(2 * α * β))
                A[ii, :] .= R * A[ii, :]
                A[:, ii] .= A[:, ii] * R'
            end
        end
        display(offdiag(A))
        iter +=1
    end
    return A
end

n = 2
W = randn(n,n)
X = randn(n,n)
W = W + W'
X = X - X'
A = [W -X; X W]
B = copy(A)
display(B[[1;3;2;4],[1;3;2;4]])
SSHjacobi!(B)
display(B)
print("ok \n")
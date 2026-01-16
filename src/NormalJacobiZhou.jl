
using LinearAlgebra
"""
```normal_jacobi_zhou!(A::AbstractMatrix)```

In-place Jacobi method for a normal matrix:
    Zhou, B., Brent, R.: An efficient method for computing eigenvalues of a real
  normal matrix. Journal of Parallel and Distributed Computing  63 (6),
   638--648 (2003), https://doi.org/10.1016/S0743-7315(03)00007-8 \\

Input:  - a normal matrix `A`.\\
Output: The real Schur form of a in a `Tridiagonal` matrix.
"""
@views function normal_jacobi_zhou!(A::AbstractMatrix)
    n = size(A, 1)
    T = typeof(A[1, 1])
    εₘ = eps(T)
    ε  = 10 * εₘ * norm(A)
    iter = 1
    itermax = 5 * n
    indices = zeros(Integer, 4)
    temp1  = zeros(T, n, 4)
    temp2 = zeros(T, 4, n)
    oldoff = Inf
    offschur = offSchur(A)
    while offschur > ε && iter < itermax && offschur < oldoff
        for i ∈ 1:2:n-2
            for j ∈ i+2:2:n-1
                indices .= i, i+1, j, j+1
                if norm(A[[j, j + 1],[i, i + 1]]) > 4εₘ
                    _, Q, v = schur(A[indices, indices])
                    if iszero(imag(v[1])) && !iszero(imag(v[2])) 
                        Base.permutecols!!(Q, [2, 3, 4, 1])
                    end
                    mul!(temp2, Q' , A[indices, :], 1, 0)
                    A[indices, :] .= temp2
                    mul!(temp1, A[:, indices], Q, 1, 0)
                    A[:, indices] .= temp1
                end
            end
        end
        oldoff = offschur
        offschur = offSchur(A)
        iter += 1
    end
    
    if iter == itermax
        @warn "Maximum number of iterations reached in normal_jacobi_zhou!"
    end

    return Tridiagonal(A)
end

normal_jacobi_zhou(A::AbstractMatrix)= normal_jacobi_zhou!(copy(A))

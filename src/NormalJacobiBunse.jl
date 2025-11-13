using LinearAlgebra
include("Utils.jl")
"""
```normal_jacobi_bunse!(A::AbstractMatrix)```

In-place Jacobi method for a normal matrix:
  Bunse-Gerstner, A., Byers, R., Mehrmann, V.: Numerical Methods for Simultaneous Diagonalization,
  SIAM J. Matrix Anal. Appl. 14 (4), 927--949, (1993),
   https://doi.org/10.1137/0614062 \\

Input:  - a normal matrix `A`.\\
Output: The real Schur form of a in a `Tridiagonal` matrix.
"""
@views function normal_jacobi_bunse!(A::AbstractMatrix)
    n = size(A, 1)
    T = typeof(A[1, 1])
    ε = 100 *eps(T) * norm(A)
    iter = 1
    itermax = 5 * sqrt(n)
    ii = zeros(Integer, 2)
    indices = zeros(Integer, 4)
    tv = zeros(T, n, 2)
    th = zeros(T, 2, n)
    th2 = zeros(T, 2, 4)
    while offSchur(A) > ε && iter < itermax
        #print("Accuracy at iter", iter, " : ", norm(A-Matrix(Tridiagonal(A))), "\n")
        for i ∈ 1:2:n-2
            for j ∈ i+2:2:n-1
                indices .= i, i+1, j, j+1
                if norm(A[[j, j + 1],[i, i + 1]]) > 100 * eps(T)
                    _, Q = schur(A[indices, indices])
                    #Compute and apply R(1,3)
                    num = Q[2,2] * Q[3, 1] - Q[2,1] * Q[3,2]
                    den = Q[1,2] * Q[2, 1] - Q[1,1] * Q[2,2] 
                    θ₁ = atan(num , den)
                    c, s = cos(θ₁), sin(θ₁)
                    ii .= 1, 3
                    th2 .= Q[ii, :]
                    @. Q[1, :] =  c * th2[1, :] + -s * th2[2, :] 
                    @. Q[3, :] =  s * th2[1, :] + c * th2[2, :]
                    ii .= i, j
                    th .= A[ii, :]
                    @. A[i, :] =  c * th[1, :] + -s * th[2, :] 
                    @. A[j, :] =  s * th[1, :] + c * th[2, :]
                    tv .= A[:, ii]
                    @. A[:, i] = c * tv[:, 1] + -s * tv[:, 2]
                    @. A[:, j] = s * tv[:, 1] + c * tv[:, 2]
                    #Compute and apply R(2,3)
                    θ₂ = atan( - Q[3, 2], Q[2, 2])
                    c, s = cos(θ₂), sin(θ₂)
                    ii .= 2, 3
                    th2 .= Q[ii, :]
                    @. Q[2, :] =  c * th2[1, :] + -s * th2[2, :] 
                    @. Q[3, :] =  s * th2[1, :] + c * th2[2, :]
                    ii .= i + 1, j
                    th .= A[ii, :]
                    @. A[i + 1, :] =  c * th[1, :] + -s* th[2, :] 
                    @. A[j, :] =  s * th[1, :] + c * th[2, :]
                    tv .= A[:, ii]
                    @. A[:, i + 1] =  c * tv[:, 1] + -s * tv[:, 2]
                    @. A[:, j] = s * tv[:, 1] + c * tv[:, 2]
                    # Compute and apply R(1,4) 
                    num = Q[2,2] * Q[4, 1] - Q[2,1] * Q[4,2]
                    den = Q[1,2] * Q[2, 1] - Q[1,1] * Q[2,2]
                    θ₃ = atan(num , den)
                    c, s = cos(θ₃), sin(θ₃)
                    ii .= 1, 4
                    th2 .= Q[ii, :]
                    @. Q[1, :] =  c * th2[1, :] + -s * th2[2, :] 
                    @. Q[4, :] =  s * th2[1, :] + c * th2[2, :]
                    ii .= i, j + 1
                    th .= A[ii, :]
                    @. A[i, :] =  c * th[1, :] + -s * th[2, :] 
                    @. A[j + 1, :] =  s * th[1, :] + c * th[2, :]
                    tv .= A[:, ii]
                    @. A[:, i] =  c * tv[:, 1] + -s * tv[:, 2]
                    @. A[:, j + 1] = s * tv[:, 1] + c * tv[:, 2]
                    # Compute and apply R(2,4)
                    θ₄ = atan( - Q[4, 2], Q[2, 2])
                    c, s = cos(θ₄), sin(θ₄)
                    ii .= 2, 4
                    th2 .= Q[ii, :]
                    @. Q[2, :] =  c * th2[1, :] + -s * th2[2, :] 
                    @. Q[4, :] =  s * th2[1, :] + c * th2[2, :]
                    ii .= i + 1, j + 1
                    th .= A[ii, :]
                    @. A[i + 1, :] =  c * th[1, :] + -s * th[2, :] 
                    @. A[j + 1, :] =  s * th[1, :] + c * th[2, :]
                    tv .= A[:, ii]
                    @. A[:, i + 1] =  c * tv[:, 1] + -s * tv[:, 2]
                    @. A[:, j + 1] = s * tv[:, 1] + c * tv[:, 2]
                    #display(Q)
                end
            end
        end
        #display(offSchur(A))
        iter += 1
    end
    #print("Accuracy at iter ", iter, " : ", norm(A-Matrix(Tridiagonal(A))), "\n")
    return Tridiagonal(A)
end

normal_jacobi_bunse(A::AbstractMatrix)= normal_jacobi_bunse!(copy(A))

n = 8
A = Matrix(qr(randn(Float64, n , n)).Q)
normal_jacobi_bunse(A)
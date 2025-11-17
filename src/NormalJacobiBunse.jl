using LinearAlgebra
include("Utils.jl")
"""
```normal_jacobi_bunse!(A::AbstractMatrix)```

In-place Jacobi method for a normal matrix:\\
  Bunse-Gerstner, A., Byers, R., Mehrmann, V.: Numerical Methods for Simultaneous Diagonalization,
  SIAM J. Matrix Anal. Appl. 14 (4), 927--949, (1993),
   https://doi.org/10.1137/0614062 \\

Input:  - a normal matrix `A`.\\
Output: The real Schur form of a in a `Tridiagonal` matrix.
"""
@views function normal_jacobi_bunse!(A::AbstractMatrix)
    n = size(A, 1)
    T = typeof(A[1, 1])
    εₘ = eps(T)
    ε = 100 * εₘ * norm(A)
    
    iter = 1
    itermax = 5 * n
    ii = zeros(Integer, 2)
    indices = zeros(Integer, 4)
    tv = zeros(T, n, 2)
    th = zeros(T, 2, n)
    th2 = zeros(T, 2, 4)
    offschur = offSchur(A)
    while offschur > ε && iter < itermax
        #print("Accuracy at iter", iter, " : ", norm(A-Matrix(Tridiagonal(A))), "\n")
        for i ∈ 1:2:n-2
            for j ∈ i+2:2:n-1
                indices .= i, i+1, j, j+1
                if norm(A[[j, j + 1],[i, i + 1]]) > 4εₘ
                    _, Q, v = schur(A[indices, indices])
                    if iszero(imag(v[1])) && !iszero(imag(v[2])) 
                        Base.permutecols!!(Q, [2, 3, 4, 1])
                    end
                    k₁, k₂, k₃, k₄ = 1, 2, 3, 4
                    i₁, i₂, j₁, j₂ = i, i + 1, j, j + 1
                    #Compute and apply R(1,3)
                    num = Q[k₂, k₂] * Q[k₃, k₁] - Q[k₂, k₁] * Q[k₃, k₂]
                    den = Q[k₁, k₂] * Q[k₂, k₁] - Q[k₁, k₁] * Q[k₂, k₂]
                    #θ₁ = atan(num , den)
                    #c, s = cos(θ₁), sin(θ₁)
                    h = hypot(num, den)
                    c = den / h
                    s = num / h
                    ii .= k₁, k₃
                    th2 .= Q[ii, :]
                    @. Q[k₁, :] =  c * th2[1, :] + -s * th2[2, :] 
                    @. Q[k₃, :] =  s * th2[1, :] + c * th2[2, :]
                    ii .= i₁, j₁
                    th .= A[ii, :]
                    @. A[i₁, :] =  c * th[1, :] + -s * th[2, :] 
                    @. A[j₁, :] =  s * th[1, :] + c * th[2, :]
                    tv .= A[:, ii]
                    @. A[:, i₁] = c * tv[:, 1] + -s * tv[:, 2]
                    @. A[:, j₁] = s * tv[:, 1] + c * tv[:, 2]
                    #Compute and apply R(2,3)
                    #θ₂ = atan( - Q[k₃, k₂], Q[k₂, k₂])
                    #c, s = cos(θ₂), sin(θ₂)
                    h = hypot(Q[k₂, k₂], - Q[k₃, k₂])
                    c = Q[k₂, k₂] / h
                    s = - Q[k₃, k₂] / h
                    ii .= k₂, k₃
                    th2 .= Q[ii, :]
                    @. Q[k₂, :] =  c * th2[1, :] + -s * th2[2, :] 
                    @. Q[k₃, :] =  s * th2[1, :] + c * th2[2, :]
                    ii .= i₂, j₁
                    th .= A[ii, :]
                    @. A[i₂, :] =  c * th[1, :] + -s* th[2, :] 
                    @. A[j₁, :] =  s * th[1, :] + c * th[2, :]
                    tv .= A[:, ii]
                    @. A[:, i₂] =  c * tv[:, 1] + -s * tv[:, 2]
                    @. A[:, j₁] = s * tv[:, 1] + c * tv[:, 2]
                    # Compute and apply R(1,4) 
                    num = Q[k₂, k₂] * Q[k₄, k₁] - Q[k₂, k₁] * Q[k₄, k₂]
                    den = Q[k₁, k₂] * Q[k₂, k₁] - Q[k₁, k₁] * Q[k₂, k₂]
                    #θ₃ = atan(num , den)
                    #c, s = cos(θ₃), sin(θ₃)
                    h = hypot(num, den)
                    c = den / h
                    s = num / h
                    ii .= k₁, k₄
                    th2 .= Q[ii, :]
                    @. Q[k₁, :] =  c * th2[1, :] + -s * th2[2, :] 
                    @. Q[k₄, :] =  s * th2[1, :] + c * th2[2, :]
                    ii .= i₁, j₂
                    th .= A[ii, :]
                    @. A[i₁, :] =  c * th[1, :] + -s * th[2, :] 
                    @. A[j₂, :] =  s * th[1, :] + c * th[2, :]
                    tv .= A[:, ii]
                    @. A[:, i₁] =  c * tv[:, 1] + -s * tv[:, 2]
                    @. A[:, j₂] = s * tv[:, 1] + c * tv[:, 2]
                    # Compute and apply R(2,4)
                    #θ₄ = atan( - Q[k₄, k₂], Q[k₂, k₂])
                    #c, s = cos(θ₄), sin(θ₄)
                    h = hypot(Q[k₂, k₂], - Q[k₄, k₂])
                    c = Q[k₂, k₂] / h
                    s = - Q[k₄, k₂] / h
                    ii .= k₂, k₄
                    th2 .= Q[ii, :]
                    @. Q[k₂, :] =  c * th2[1, :] + -s * th2[2, :] 
                    @. Q[k₄, :] =  s * th2[1, :] + c * th2[2, :]
                    ii .= i₂, j₂
                    th .= A[ii, :]
                    @. A[i₂, :] =  c * th[1, :] + -s * th[2, :] 
                    @. A[j₂, :] =  s * th[1, :] + c * th[2, :]
                    tv .= A[:, ii]
                    @. A[:, i₂] =  c * tv[:, 1] + -s * tv[:, 2]
                    @. A[:, j₂] = s * tv[:, 1] + c * tv[:, 2]
                    
                end
            end
        end
        offschur = offSchur(A)
        #display(offschur)
        iter += 1
    end
    return Tridiagonal(A)
end

normal_jacobi_bunse(A::AbstractMatrix)= normal_jacobi_bunse!(copy(A))
#=
n = 100
A = Matrix(qr(randn(n, n)).Q)
if det(A) < 0
    #[:, 1] .= -A[:, 1]
end
H = (A + A') / 2
Ω = (A - A') / 2
A = H + 0.2 * Ω
normal_jacobi_bunse!(copy(A))
print("Done Bunse\n")
=#

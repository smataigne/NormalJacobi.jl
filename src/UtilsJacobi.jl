using LinearAlgebra

include("Utils.jl")
"""
```normal_jacobi_bunse2!(A::AbstractMatrix)```

In-place Jacobi method for a normal matrix:\\
  Bunse-Gerstner, A., Byers, R., Mehrmann, V.: Numerical Methods for Simultaneous Diagonalization,
  SIAM J. Matrix Anal. Appl. 14 (4), 927--949, (1993),
   https://doi.org/10.1137/0614062 \\

Input:  - a normal matrix `A`.\\
Output: The real Schur form of a in a `Tridiagonal` matrix.
"""
@views function normal_jacobi_bunse2!(A::AbstractMatrix, kk::Vector{Int})
    n = size(A, 1)
    T = typeof(A[1, 1])
    εₘ = eps(T)
    ε = 10 * εₘ * norm(A[kk, kk])
    nk = length(kk)
    iter = 1
    itermax = 5 * nk
    ii = zeros(Integer, 2)
    indices = zeros(Integer, 4)
    tv = zeros(T, n, 2)
    th = zeros(T, 2, n)
    th2 = zeros(T, 2, 4)
    oldoff = Inf
    offschur = offSchur(A[kk, kk])
    while offschur > ε && iter < itermax && offschur < oldoff
        #print("Accuracy at iter", iter, " : ", norm(A-Matrix(Tridiagonal(A))), "\n")
        for i ∈ 1:2:nk-2
            for j ∈ i+2:2:nk-1
                indices .= kk[i], kk[i+1], kk[j], kk[j+1]
                if norm(A[[kk[j], kk[j + 1]],[kk[i], kk[i + 1]]]) > 4εₘ
                    _, Q, v = schur(A[indices, indices])
                    if iszero(imag(v[1])) && !iszero(imag(v[2])) 
                        Base.permutecols!!(Q, [2, 3, 4, 1])
                    end
                    k₁, k₂, k₃, k₄ = 1, 2, 3, 4
                    i₁, i₂, j₁, j₂ = kk[i], kk[i+1], kk[j], kk[j+1]
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
        oldoff = offschur
        offschur = offSchur(A[kk, kk])
        #display(offschur)
        iter += 1
    end

    if iter == itermax
        @warn "Maximum number of iterations reached in normal_jacobi_bunse!"
    end

    return Tridiagonal(A)
end

"""
    SSHjacobi2!(A::AbstractMatrix{T}) where T

In-place Symmetric Skew-Hamiltonian Jacobi method for a symmetric matrix A of even size.
"""
@views function SSHjacobi2!(A::AbstractMatrix{T}, kk::AbstractVector{Int} ) where T
    n = size(A, 1)
    itermax = 10; iter = 1
    nk = length(kk)
    ii = zeros(Int, 4)
    p = zeros(T, 3)
    R = zeros(T, 4, 4)
    temp1 = zeros(T, 4, n)
    temp2 = zeros(T, n, 4)
    ε = eps(T) * 100 * norm(A)
    oldoff = Inf
    offdiagA = offSchur(A[kk, kk])
    while offdiagA > ε && iter < itermax && offdiagA < oldoff
        for i ∈ 1:2:nk-3
            for j ∈ i+2:2:nk-1
                #ii .= i, i+1, j, j+1
                ii .= kk[i], kk[i+1], kk[j], kk[j+1]
                #Take the symmetric skew-Hamiltonian part
                w₁ = 0.5 * (A[ii[1], ii[1]] + A[ii[2], ii[2]])
                w₂ = 0.25 * (A[ii[1], ii[3]] + A[ii[3], ii[1]] + A[ii[2], ii[4]] + A[ii[4], ii[2]])
                w₃ = 0.5 * (A[ii[3], ii[3]] + A[ii[4], ii[4]])
                x = 0.25 * (A[ii[1], ii[4]] - A[ii[2], ii[3]] - A[ii[3], ii[2]] + A[ii[1], ii[4]])
                p[1] = -x
                p[2] = 0.5 * (w₁ - w₃)
                p[3] = w₂
                #p[1] = -A[ii[1], ii[4]]
                #p[2] = 0.5 * (A[ii[1], ii[1]] - A[ii[3], ii[3]])
                #p[3] = A[ii[1], ii[3]]
                α = norm(p)
                β = α + p[2]
                R[:, 1] .= β, 0.0, -p[3], p[1]
                R[:, 3] .= p[3], p[1], β, 0.
                R[:, 2] .= 0. , β, -p[1], -p[3]
                R[:, 4] .= -p[1], p[3], 0.0, β
                R .*= (1 / √(2 * α * β))
                temp1 .= A[ii, :]
                mul!(A[ii, :], R, temp1, 1, 0)
                temp2 .= A[:, ii]
                mul!(A[:, ii], temp2, R', 1, 0)
            end
        end
        iter +=1
        oldoff = offdiagA
        offdiagA = offSchur(A[kk, kk])
    end
    return A
end

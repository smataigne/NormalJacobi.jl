using LinearAlgebra
include("Utils.jl")

@views function implicit_paardekooper!(A::AbstractMatrix{T}) where T
    n  = size(A, 1)
    εₘ = eps(T)                  #Element-wise norm bound
    ε  = εₘ * norm(A) * 10       #Matrix-wise norm bound
    ii = zeros(Int64, 2)         #Indices for rows/columns selections
    th = zeros(T, 2, n)
    tv = zeros(T, n, 2)
    iter = 1
    itermax = 5 * sqrt(n)
    oldoff = Inf
    offschur = skew_offschur(A)
    #Phase I  Implicit Paardekooper
    while offschur > ε && iter < itermax && offschur < oldoff
        for i ∈ 1:2:n-3
            for j ∈ i+2:2:n-1
                #First Jacobi Annihilator
                #c₁, s₁, c₂, s₂ = annihilator(Ω[i + 1, i], Ω[i + 1, j], Ω[j + 1, i], Ω[j + 1, j])
                ω1 = (A[i + 1, i] - A[i, i + 1]) / 2
                ω2 = (A[i + 1, j] - A[j, i + 1]) / 2
                ω3 = (A[j + 1, i] - A[i, j + 1]) / 2
                ω4 = (A[j + 1, j] - A[j, j + 1]) / 2
                if abs(ω2) + abs(ω3) > 2εₘ
                    c₁, s₁, c₂, s₂ = annihilator(ω1, ω2, ω3, ω4)
                    #First Similarity transformation "G1' * A * G1"
                    #G1 = [c₁ -s₁; s₁ c₁]
                    #A[[i+1, j+1], :] = G1'A[[i+1, j+1], :]
                    #A[:, [i+1, j+1]] = A[:, [i+1, j+1]] * G1
                    ii .= i + 1, j + 1
                    th .= A[ii, :]
                    @. A[i + 1, :] =  c₁ * th[1, :] + s₁ * th[2, :] 
                    @. A[j + 1, :] = -s₁ * th[1, :] + c₁ * th[2, :] 
                    tv .= A[:, ii]
                    @. A[:, i + 1] =  c₁ * tv[:, 1] + s₁ * tv[:, 2]
                    @. A[:, j + 1] = -s₁ * tv[:, 1] + c₁ * tv[:, 2]

                    #Second Similarity transformation "G2' * A * G2"
                    #G2 = [c₂ -s₂; s₂ c₂]
                    #A[:, [i, j]] = A[:, [i, j]] * G2
                    #A[[i, j], :] = G2'A[[i, j], :]
                    ii .= i, j
                    tv .= A[:, ii]
                    @. A[:, i] =  c₂ * tv[:, 1] + s₂ * tv[:, 2]
                    @. A[:, j] = -s₂ * tv[:, 1] + c₂ * tv[:, 2]
                    th .= A[ii, :]
                    @. A[i, :] =  c₂ * th[1, :] + s₂ * th[2, :] 
                    @. A[j, :] = -s₂ * th[1, :] + c₂ * th[2, :] 
                end
                #c₁, s₁, c₂, s₂ = annihilator(Ω[i + 1, i], Ω[i + 1, j + 1], Ω[j, i], Ω[j, j + 1])
                ω1 = (A[i + 1, i] - A[i, i + 1]) / 2
                ω2 = (A[i + 1, j + 1] - A[j + 1, i + 1]) / 2
                ω3 = (A[j, i] - A[i, j]) / 2
                ω4 = (A[j, j + 1] - A[j + 1, j]) / 2
                if abs(ω2) + abs(ω3) > 2εₘ     
                    c₁, s₁, c₂, s₂ = annihilator(ω1, ω2, ω3, ω4)
                    #First Similarity transformation "G1' * A * G1"
                    #G1 = [c₁ -s₁; s₁ c₁]
                    ii .= i + 1, j
                    th .= A[ii, :]
                    @. A[i + 1, :] =  c₁ * th[1, :] + s₁ * th[2, :] 
                    @. A[j, :] = -s₁ * th[1, :] + c₁ * th[2, :] 
                    tv .= A[:, ii]
                    @. A[:, i + 1] =  c₁ * tv[:, 1] + s₁ * tv[:, 2]
                    @. A[:, j] = -s₁ * tv[:, 1] + c₁ * tv[:, 2]
                    #Second Similarity transformation "G2' * A * G2"
                    #G2 = [c₂ -s₂; s₂ c₂]
                    ii .= i, j + 1
                    tv .= A[:, ii]
                    @. A[:, i] =  c₂ * tv[:, 1] + s₂ * tv[:, 2]
                    @. A[:, j + 1] = -s₂ * tv[:, 1] + c₂ * tv[:, 2]
                    th .= A[ii, :]
                    @. A[i, :] =  c₂ * th[1, :] + s₂ * th[2, :] 
                    @. A[j + 1, :] = -s₂ * th[1, :] + c₂ * th[2, :]
                end
            end
        end
        oldoff = offschur
        offschur = skew_offschur(A)
        iter += 1
    end
    if offschur > 10ε
        @warn "Phase I did not converge to the desired accuracy!"
    end
    #Ensures correct signs on Ω
    for i ∈ 1:n-1
        if (A[i + 1, i] - A[i, i + 1]) < 0
            A[:, i + 1] .*= -1
            A[i + 1, :] .*= -1
        end
    end
    return A
end

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
    offschur = offschur(A[kk, kk])
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
        offschur = offschur(A[kk, kk])
        #display(offschur)
        iter += 1
    end

    if iter == itermax
        @warn "Maximum number of iterations reached in normal_jacobi_bunse!"
    end

    return Tridiagonal(A)
end

"""
    ssh_jacobi2!(A::AbstractMatrix{T}) where T

In-place Symmetric Skew-Hamiltonian Jacobi method for a symmetric matrix A of even size.
"""
@views function ssh_jacobi2!(A::AbstractMatrix{T}, kk::AbstractVector{Int} ) where T
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
    offdiagA = ssh_offdiag(A[kk, kk])
    while offdiagA > ε && iter < itermax && offdiagA < oldoff
        for i ∈ 1:2:nk-3
            for j ∈ i+2:2:nk-1
                #ii .= i, i+1, j, j+1
                ii .= kk[i], kk[i+1], kk[j], kk[j+1]
                #Take the symmetric skew-Hamiltonian part
                w₁ = 0.5 * (A[ii[1], ii[1]] + A[ii[2], ii[2]])
                w₂ = 0.25 * (A[ii[1], ii[3]] + A[ii[3], ii[1]] + A[ii[2], ii[4]] + A[ii[4], ii[2]])
                w₃ = 0.5 * (A[ii[3], ii[3]] + A[ii[4], ii[4]])
                x = 0.25 * (A[ii[1], ii[4]] - A[ii[2], ii[3]] - A[ii[3], ii[2]] + A[ii[4], ii[1]])
                p[1] = -x
                p[2] = 0.5 * (w₁ - w₃)
                p[3] = w₂
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
        offdiagA = ssh_offdiag(A[kk, kk])
    end
    return A
end

@views function symmetric_jacobi2!(A::AbstractMatrix{T}, kk::AbstractVector{Int} ) where T
    K = length(kk)
    n = size(A, 1)
    iter = 1; itermax = max(5 * sqrt(K), 15)
    ii = zeros(Int, 2)
    th = zeros(T, 2, n)
    tv = zeros(T, n, 2)
    old_offdiag = Inf
    new_offdiag = sym_offdiag(A[kk, kk])
    ε = eps(T) * 10 * norm(A)
    εₘ = eps(T) * 10
    while new_offdiag > ε && iter < itermax && new_offdiag < old_offdiag
        for i ∈ 1:K-1
            for j ∈ i+1:K
                r = (A[kk[j], kk[i]] + A[kk[j], kk[i]]) / 2
                if abs(r) > εₘ
                    c, s = jacobi_rotation(A[kk[i], kk[i]], r, A[kk[j], kk[j]])
                    ii .= kk[i], kk[j]
                    th .= A[ii, :]
                    @. A[ii[1], :] =  c * th[1, :] + s * th[2, :] 
                    @. A[ii[2], :] = -s * th[1, :] + c * th[2, :] 
                    tv .= A[:, ii]
                    @. A[:, ii[1]] =  c * tv[:, 1] + s * tv[:, 2]
                    @. A[:, ii[2]] = -s * tv[:, 1] + c * tv[:, 2]
                end
            end
        end
        iter += 1
        old_offdiag = new_offdiag
        new_offdiag = sym_offdiag(A[kk, kk])
    end
    return A
end
#=
m = 6
H = randn(m ,m)
H[1,2] *= 1e-6
H[2,1] *= 1e-6
H .+= H'
Ω = 1e-6 * randn(m, m)
Ω .-= Ω'
A = [H -Ω; Ω H]
H1 = randn(m ,m)
H1 .+= H1'
Ω1 = randn(m, m)
Ω1 .-= Ω1'
P = randn(2m, 2m)#[Ω1 H1; -H1 Ω1]
display(tr(P'A))
display(offdiagssh(P))
A .+= 1e-12 * P
kk = invpermute!(Array(1:2m), [1:2:2m;2:2:2m])
display(isSSH(A[kk, kk], 1e-4))
display(isSSH(A[kk, kk], 1e-12))
M = A[kk, kk]
SSHjacobi2!(M, Array(1:2m))
display(sshpart(M))
#=
A = [Ω H; -H Ω]
offdiagssh(A[kk, kk])
=#
#display(normofssh(A[kk, kk]))
=#


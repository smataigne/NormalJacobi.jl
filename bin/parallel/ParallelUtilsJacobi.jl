using LinearAlgebra
include("Utils.jl")


#works for size of multiples of 4n only

@views function parallel_implicit_paardekooper!(A::AbstractMatrix{T}, temp::AbstractMatrix) where T
    n  = size(A, 1)
    n2 = n ÷ 2
    εₘ = eps(T)                  #Element-wise norm bound
    ε  = εₘ * norm(A) * 10       #Matrix-wise norm bound
            #Indices for rows/columns selections
    iter = 1
    itermax = 5 * sqrt(n)
    oldoff = Inf
    new_offschur = skew_offschur(A)
    steps = parallel_cyclic_order(n2)
    do_it = zeros(Bool, n) 
    cs1 = zeros(T, n)
    cs2 = zeros(T, n)
    ss1 = zeros(T, n)
    ss2 = zeros(T, n)
    #Phase I  Implicit Paardekooper
    while new_offschur > ε && iter < itermax
        if oldoff < new_offschur && new_offschur < 10ε
            @warn "Paardekooper's threshold was too low"
            break
        end
        for k1 ∈ 1:(n2 - 1)
            Threads.@threads for k2 ∈ 1:(n2 ÷ 2)
                i = 2 * minimum(steps[k1, k2, :]) - 1
                j = 2 * maximum(steps[k1, k2, :]) - 1
                #First Jacobi Annihilator
                #c₁, s₁, c₂, s₂ = annihilator(Ω[i + 1, i], Ω[i + 1, j], Ω[j + 1, i], Ω[j + 1, j])
                ω1 = (A[i + 1, i] - A[i, i + 1]) / 2
                ω2 = (A[i + 1, j] - A[j, i + 1]) / 2
                ω3 = (A[j + 1, i] - A[i, j + 1]) / 2
                ω4 = (A[j + 1, j] - A[j, j + 1]) / 2
                if abs(ω2) + abs(ω3) > 2εₘ
                    do_it[k2] = true
                    c₁, s₁, c₂, s₂ = annihilator(ω1, ω2, ω3, ω4)
                    cs1[k2] = c₁
                    ss1[k2] = s₁
                    cs2[k2] = c₂ 
                    ss2[k2] = s₂ 
                    #First Similarity transformation "G1' * A * G1"
                    ii = zeros(Int64, 2) 
                    ii .= i + 1, j + 1
                    temp[ii, :] .= A[ii, :]
                    @. A[i + 1, :] =  c₁ * temp[i + 1, :] + s₁ * temp[j + 1, :] 
                    @. A[j + 1, :] = -s₁ * temp[i + 1, :] + c₁ * temp[j + 1, :] 
                    ii .= i, j
                    temp[ii, :] .= A[ii, :]
                    @. A[i, :] =  c₂ * temp[i, :] + s₂ * temp[j, :] 
                    @. A[j, :] = -s₂ * temp[i, :] + c₂ * temp[j, :] 
                end
            end
            Threads.@threads for k2 ∈ 1:(n2 ÷ 2)
                i = 2 * minimum(steps[k1, k2, :]) - 1
                j = 2 * maximum(steps[k1, k2, :]) - 1
                if do_it[k2]
                    #Second Similarity transformation "G2' * A * G2"
                    ii = zeros(Int64, 2) 
                    ii .= i + 1, j + 1
                    temp[:, ii] .= A[:, ii]
                    @. A[:, i + 1] =  cs1[k2] * temp[:, i + 1] + ss1[k2] * temp[:, j + 1]
                    @. A[:, j + 1] = -ss1[k2] * temp[:, i + 1] + cs1[k2] * temp[:, j + 1]
                    ii .= i, j
                    temp[:, ii] .= A[:, ii]
                    @. A[:, i] =  cs2[k2] * temp[:, i] + ss2[k2] * temp[:, j]
                    @. A[:, j] = -ss2[k2] * temp[:, i] + cs2[k2] * temp[:, j]
                    do_it[k2] = false
                end
            end
            Threads.@threads for k2 ∈ 1:(n2 ÷ 2)
                i = 2 * minimum(steps[k1, k2, :]) - 1
                j = 2 * maximum(steps[k1, k2, :]) - 1
                #First Jacobi Annihilator
                #c₁, s₁, c₂, s₂ = annihilator(Ω[i + 1, i], Ω[i + 1, j], Ω[j + 1, i], Ω[j + 1, j])
                ω1 = (A[i + 1, i] - A[i, i + 1]) / 2
                ω2 = (A[i + 1, j + 1] - A[j + 1, i + 1]) / 2
                ω3 = (A[j, i] - A[i, j]) / 2
                ω4 = (A[j, j + 1] - A[j + 1, j]) / 2
                if abs(ω2) + abs(ω3) > 2εₘ
                    do_it[k2] = true
                    c₁, s₁, c₂, s₂ = annihilator(ω1, ω2, ω3, ω4)
                    cs1[k2] = c₁
                    ss1[k2] = s₁
                    cs2[k2] = c₂ 
                    ss2[k2] = s₂ 
                    #First Similarity transformation "G1' * A * G1"
                    ii = zeros(Int64, 2) 
                    ii .= i + 1, j
                    temp[ii, :] .= A[ii, :]
                    @. A[i + 1, :] =  c₁ * temp[i + 1, :] + s₁ * temp[j, :] 
                    @. A[j, :] = -s₁ * temp[i + 1, :] + c₁ * temp[j, :] 
                    ii .= i, j + 1
                    temp[ii, :] .= A[ii, :]
                    @. A[i, :] =  c₂ * temp[i, :] + s₂ * temp[j + 1, :] 
                    @. A[j + 1, :] = -s₂ * temp[i, :] + c₂ * temp[j + 1, :] 
                end
            end
            Threads.@threads for k2 ∈ 1:(n2 ÷ 2)
                i = 2 * minimum(steps[k1, k2, :]) - 1
                j = 2 * maximum(steps[k1, k2, :]) - 1
                if do_it[k2]
                    #Second Similarity transformation "G2' * A * G2"
                    ii = zeros(Int64, 2) 
                    ii .= i + 1, j
                    temp[:, ii] .= A[:, ii]
                    @. A[:, i + 1] =  cs1[k2] * temp[:, i + 1] + ss1[k2] * temp[:, j]
                    @. A[:, j] = -ss1[k2] * temp[:, i + 1] + cs1[k2] * temp[:, j]
                    ii .= i, j + 1
                    temp[:, ii] .= A[:, ii]
                    @. A[:, i] =  cs2[k2] * temp[:, i] + ss2[k2] * temp[:, j + 1]
                    @. A[:, j + 1] = -ss2[k2] * temp[:, i] + cs2[k2] * temp[:, j + 1]
                    do_it[k2] = false
                end
            end
        end
        oldoff = new_offschur
        new_offschur = skew_offschur(A)
        iter += 1
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
@views function parallel_normal_jacobi_bunse2!(A::AbstractMatrix, kk::Vector{Int}, temp::AbstractMatrix)
    n = size(A, 1)
    T = typeof(A[1, 1])
    εₘ = eps(T)
    ε = 10 * εₘ * norm(A[kk, kk])
    nk = length(kk)
    nk2 = nk ÷ 2
    iter = 1
    itermax = 5 * nk
    
    Qs = zeros(4, n)
    old_offschur = Inf
    new_offschur = offschur(A[kk, kk])
    steps = parallel_cyclic_order(nk2)
    do_it = zeros(Bool, n)
    while new_offschur > ε && iter < itermax && new_offschur < old_offschur
        for k1 ∈ 1:nk2-1
            Threads.@threads for k2 ∈ 1:(nk2 ÷ 2)
                i = 2 * min(steps[k1, k2, 1], steps[k1, k2, 2]) - 1
                j = 2 * max(steps[k1, k2, 1], steps[k1, k2, 2]) - 1
                indices = zeros(Integer, 4)
                indices .= kk[i], kk[i+1], kk[j], kk[j+1]
                if norm(A[[kk[j], kk[j + 1]],[kk[i], kk[i + 1]]]) > 4εₘ
                    do_it[kk[i]] = true
                    _, Q, v = schur(A[indices, indices])
                    Qs[:, indices] .= Q
                    if iszero(imag(v[1])) && !iszero(imag(v[2])) 
                        Base.permutecols!!(Q, [2, 3, 4, 1])
                    end
                    if iszero(imag(v[1])) && !iszero(imag(v[2])) 
                        Base.permutecols!!(Q, [2, 3, 4, 1])
                    end
                    mul!(temp[indices, :], Q' , A[indices, :], 1, 0)
                    A[indices, :] .= temp[indices, :]
                end
            end
            Threads.@threads for k2 ∈ 1:(nk2 ÷ 2)
                i = 2 * min(steps[k1, k2, 1], steps[k1, k2, 2]) - 1
                j = 2 * max(steps[k1, k2, 1], steps[k1, k2, 2]) - 1
                indices = zeros(Integer, 4)
                indices .= kk[i], kk[i+1], kk[j], kk[j+1]
                if do_it[kk[i]]
                    mul!(temp[:, indices], A[:, indices], Qs[:, indices], 1, 0)
                    A[:, indices] .= temp[:, indices]
                    do_it[kk[i]] = false
                end 
            end
        end
        old_offschur = new_offschur
        new_offschur = offschur(A[kk, kk])
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
@views function parallel_ssh_jacobi2!(A::AbstractMatrix{T}, kk::AbstractVector{Int}, temp::AbstractMatrix) where T
    n = size(A, 1)
    itermax = 10; iter = 1
    nk = length(kk)
    nk2 = nk ÷ 2
    R = zeros(T, 4, n)
    p = zeros(T, 3)
    #R = zeros(T, 4, 4)
    ε = eps(T) * 100 * norm(A)
    oldoff = Inf
    offdiagA = ssh_offdiag(A[kk, kk])
    steps = parallel_cyclic_order(nk2)
    do_it = zeros(Bool, n)
    while offdiagA > ε && iter < itermax && offdiagA < oldoff
         for k1 ∈ 1:nk2-1
            Threads.@threads for k2 ∈ 1:(nk2 ÷ 2)
                i = 2 * min(steps[k1, k2, 1], steps[k1, k2, 2]) - 1
                j = 2 * max(steps[k1, k2, 1], steps[k1, k2, 2]) - 1
                #ii = zeros(Int, 4)
                i1, i2, i3, i4 .= kk[i], kk[i+1], kk[j], kk[j+1]
                #Take the symmetric skew-Hamiltonian part
                w₁ = 0.5 * (A[i1, i1] + A[i2, i2])
                w₂ = 0.25 * (A[i1, i3] + A[i3, i1] + A[i2, i4] + A[i4, i2])
                w₃ = 0.5 * (A[i3, i3] + A[i4, i4])
                x = 0.25 * (A[i1, i4] - A[i2, i3] - A[i3, i2] + A[i4, i1])
                p[1] = -x
                p[2] = 0.5 * (w₁ - w₃)
                p[3] = w₂
                α = norm(p)
                β = α + p[2]
                r = 
                R[:, i1] .= β, 0.0, -p[3], p[1]
                R[:, i3] .= p[3], p[1], β, 0.
                R[:, i2] .= 0. , β, -p[1], -p[3]
                R[:, i4] .= -p[1], p[3], 0.0, β
                R[:, i1:i4] .*= (1 / √(2 * α * β))
                ii = [i1, i2, i3, i4]
                temp[ii, :] .= A[ii, :]
                mul!(A[ii, :], R[:, ii], temp[ii, :], 1, 0)
            end
            Threads.@threads for k2 ∈ 1:(nk2 ÷ 2) 
                i = 2 * min(steps[k1, k2, 1], steps[k1, k2, 2]) - 1
                j = 2 * max(steps[k1, k2, 1], steps[k1, k2, 2]) - 1
                #ii = zeros(Int, 4)
                i1, i2, i3, i4 .= kk[i], kk[i+1], kk[j], kk[j+1]
                ii = [i1, i2, i3, i4]
                temp[:, ii] .= A[:, ii]
                mul!(A[:, ii], temp[:, ii], R[:, ii]', 1, 0)
            end
        end
        iter +=1
        oldoff = offdiagA
        offdiagA = ssh_offdiag(A[kk, kk])
        #display(offdiagA)
    end
    return A
end

@views function parallel_symmetric_jacobi2!(A::AbstractMatrix{T}, kk::AbstractVector{Int}, temp::AbstractMatrix) where T
    K = length(kk)
    n = size(A, 1)
    iter = 1
    itermax = min(5 * sqrt(K), 20)
    old_offdiag = Inf
    new_offdiag = sym_offdiag(A[kk, kk])
    ε = eps(T) * 10 * norm(A)
    εₘ = eps(T) * 10
    steps = parallel_cyclic_order(K)
    do_it = zeros(Bool, n)
    cs = zeros(n)
    ss = zeros(n)
    while new_offdiag > ε && iter < itermax && new_offdiag < old_offdiag
        for k1 ∈ 1:K-1
            Threads.@threads for k2 ∈ 1:(K ÷ 2)
                i = min(steps[k1, k2, 1], steps[k1, k2, 2])
                j = max(steps[k1, k2, 1], steps[k1, k2, 2])
                ii = kk[i]
                jj = kk[j]
                r = (A[jj, ii] + A[ii, jj]) / 2
                if abs(r) > εₘ
                    do_it[ii] = true
                    c, s = jacobi_rotation(A[ii, ii], r, A[jj, jj])
                    cs[ii] = c
                    ss[ii] = s
                    temp[ii, :] .= A[ii, :]
                    temp[jj, :] .= A[jj, :]
                    @. A[ii, :] =  cs[ii] * temp[ii, :] + ss[ii] * temp[jj, :]
                    @. A[jj, :] = -ss[ii] * temp[ii, :] + cs[ii] * temp[jj, :]
                else
                    do_it[ii] = false
                end
            end
            
            Threads.@threads for k2 ∈ 1:(K ÷ 2)
                i = min(steps[k1, k2, 1], steps[k1, k2, 2])
                j = max(steps[k1, k2, 1], steps[k1, k2, 2])
                ii = kk[i]
                jj = kk[j]
                if do_it[ii]
                    temp[:, ii] .= A[:, ii]
                    temp[:, jj] .= A[:, jj]
                    @. A[:, ii] =  cs[ii] * temp[:, ii] + ss[ii] * temp[:, jj]
                    @. A[:, jj] = -ss[ii] * temp[:, ii] + cs[ii] * temp[:, jj]
                    do_it[ii] = false
                end
            end
        end
        iter += 1
        old_offdiag = new_offdiag
        new_offdiag = sym_offdiag(A[kk, kk])
        #(new_offdiag)
    end
    return A
end


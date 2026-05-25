using LinearAlgebra
include("Utils.jl")

"""
    normal_jacobi_He!(A::AbstractMatrix; full::Bool = false)

Cyclic Jacobi-like implementation of Randdiag for normal matrices from\

    He, H., Kressner, D. A simple, randomized algorithm for diagonalizing normal matrices. Calcolo 62, 30 (2025). https://doi.org/10.1007/s10092-025-00654-z

Input:  A normal matrix `A`.\\
Output: The matrix `A` diagonalized.
"""
@views function normal_jacobi_he!(A::AbstractMatrix{T}) where T
    n = size(A, 1)
    if T<:Real
        εₘ = eps(T)
        Ac = zeros(Complex{T}, n, n)
        @. Ac = complex(A, 0)
        thc = zeros(Complex{T}, 2, n)
        tvc = zeros(Complex{T}, n, 2)
    else
        εₘ = eps(real(T))
        Ac = A
        thc = zeros(T, 2, n)
        tvc = zeros(T, n, 2)
    end
    ε  =  10 * εₘ * norm(A)
    μ₁ = randn()
    μ₂ = randn()
    ii = zeros(Integer, 2)
    iter = 1; itermax = max(5 * sqrt(n), 15)
    old_offdiag = Inf  #Randdiag doesn't converge to full precision.
    new_offdiag = offdiag(Ac)
    while new_offdiag > ε && iter < itermax && new_offdiag < old_offdiag
        for i ∈ 1:n-1
            for j ∈ i+1:n
                if abs(Ac[j, i]) > εₘ
                    ii .= i, j 
                    #Compute the Jacobi rotation for Hermitian H = μ₁ * (A + A') / 2 + 1im * μ₂ * (A - A') / 2
                    h11 = real(μ₁ * (Ac[i, i] + Ac[i, i]') / 2 + 1im * μ₂ * (Ac[i, i] - Ac[i, i]') / 2)
                    h22 = real(μ₁ * (Ac[j, j] + Ac[j, j]') / 2 + 1im * μ₂ * (Ac[j, j] - Ac[j, j]') / 2)
                    h21 = μ₁ * (Ac[j, i] + Ac[i, j]') / 2 + 1im * μ₂ * (Ac[j, i] - Ac[i, j]') / 2 
                    c, s = jacobi_rotation(h11, abs(h21), h22)
                    sα, cα = sincos(angle(h21))
                    s = complex(cα, sα) * s
                    #Apply the Jacobi rotation to A
                    ii .= i, j
                    tvc .= Ac[:, ii]
                    @. Ac[:, i] =  c * tvc[:, 1] + s * tvc[:, 2]
                    @. Ac[:, j] = -s'* tvc[:, 1] + c * tvc[:, 2]
                    thc .= Ac[ii, :]
                    @. Ac[i, :] =  c * thc[1, :] + s'* thc[2, :] 
                    @. Ac[j, :] = -s * thc[1, :] + c * thc[2, :] 
                end
            end
        end
        iter += 1
        old_offdiag = new_offdiag
        new_offdiag = offdiag(Ac)
    end

    if iter == itermax
        @warn "Maximum number of iterations reached in normal_jacobi_he!"
    end
    
    return Ac   
end

@views function parallel_normal_jacobi_he!(A::AbstractMatrix{T}) where T
    n = size(A, 1)
    if T<:Real
        εₘ = eps(T)
        Ac = zeros(Complex{T}, n, n)
        @. Ac = complex(A, 0)
        thc = zeros(Complex{T}, 2, n)
        tvc = zeros(Complex{T}, n, 2)
    else
        εₘ = eps(real(T))
        Ac = A
        thc = zeros(T, 2, n)
        tvc = zeros(T, n, 2)
    end
    ε  =  10 * εₘ * norm(A)
    μ₁ = randn()
    μ₂ = randn()
    ii = zeros(Integer, 2)
    iter = 1; itermax = max(5 * sqrt(n), 15)
    old_offdiag = Inf  #Randdiag doesn't converge to full precision.
    new_offdiag = offdiag(Ac)
    steps = parallel_cyclic_order(n)
    while new_offdiag > ε && iter < itermax && new_offdiag < old_offdiag
        for pairs ∈ steps
            Threads.@threads for pair in pairs
                i = minimum(pair)
                j = maximum(pair) 
                #for i ∈ 1:n-1
                #for j ∈ i+1:n
                if abs(Ac[j, i]) > εₘ
                    ii .= i, j 
                    #Compute the Jacobi rotation for Hermitian H = μ₁ * (A + A') / 2 + 1im * μ₂ * (A - A') / 2
                    h11 = real(μ₁ * (Ac[i, i] + Ac[i, i]') / 2 + 1im * μ₂ * (Ac[i, i] - Ac[i, i]') / 2)
                    h22 = real(μ₁ * (Ac[j, j] + Ac[j, j]') / 2 + 1im * μ₂ * (Ac[j, j] - Ac[j, j]') / 2)
                    h21 = μ₁ * (Ac[j, i] + Ac[i, j]') / 2 + 1im * μ₂ * (Ac[j, i] - Ac[i, j]') / 2 
                    c, s = jacobi_rotation(h11, abs(h21), h22)
                    sα, cα = sincos(angle(h21))
                    s = complex(cα, sα) * s
                    #Apply the Jacobi rotation to A
                    ii .= i, j
                    tvc .= Ac[:, ii]
                    @. Ac[:, i] =  c * tvc[:, 1] + s * tvc[:, 2]
                    @. Ac[:, j] = -s'* tvc[:, 1] + c * tvc[:, 2]
                    thc .= Ac[ii, :]
                    @. Ac[i, :] =  c * thc[1, :] + s'* thc[2, :] 
                    @. Ac[j, :] = -s * thc[1, :] + c * thc[2, :] 
                end
            end
        end
        iter += 1
        old_offdiag = new_offdiag
        new_offdiag = offdiag(Ac)
    end

    if iter == itermax
        @warn "Maximum number of iterations reached in normal_jacobi_he!"
    end
    
    return Ac   
end
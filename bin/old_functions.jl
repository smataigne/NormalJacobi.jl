"""
    jacobi_ssh!(A::AbstractMatrix{T}) where T

In-place Symmetric Skew-Hamiltonian Jacobi method for a symmetric matrix A of even size.
"""
@views function jacobi_ssh!(A::AbstractMatrix{T}) where T
    n = size(A, 1)
    itermax = 10; iter = 1
    n2 = n ÷ 2
    ii = zeros(Int, 4)
    p = zeros(T, 3)
    R = zeros(T, 4, 4)
    temp1 = zeros(T, 4, n)
    temp2 = zeros(T, n, 4)
    ε = eps(T) * 100 * norm(A)
    oldoff = Inf
    offdiagA = offdiag(A)
    while offdiagA > ε && iter < itermax && offdiagA < oldoff
        for i ∈ 1:n2-1
            for j ∈ i+1:n2
                ii .= i, j, i+n2, j+n2
                #M = A[ii, ii]
                #p .= -M[1, 4], 0.5*(M[1, 1] - M[2, 2]), M[1, 2]
                p[1] = -A[i, j + n2]
                p[2] = 0.5 * (A[i, i] - A[j, j])
                p[3] = A[i, j]
                α = norm(p)
                β = α + p[2]
                R[:, 1] .= β, -p[3], 0. , p[1]
                R[:, 2] .= p[3], β, p[1], 0.
                R[:, 3] .= 0. , -p[1], β, -p[3]
                R[:, 4] .= -p[1], 0. , p[3], β
                R .*= (1 / √(2 * α * β))
                temp1 .= A[ii, :]
                mul!(A[ii, :], R, temp1, 1, 0)
                temp2 .= A[:, ii]
                mul!(A[:, ii], temp2, R', 1, 0)
            end
        end
        iter +=1
        oldoff = offdiagA
        offdiagA = offdiag(A)
    end
    return A
end

"""
    findzeros!(Σ::AbstractVector{T}) where T
    
Find the indices of the (approximate) zeros in the vector Σ.
"""
function findzeros!(Σ::AbstractVector{T}, ε::Number) where T
    n = length(Σ)
    zeros_indices = zeros(Int, n + 1)  #s₀ in the paper
    i = 1; count = 0
    maxval = 0.0
    τ₀ = Inf
    for i ∈ 1:2:n
        if abs(Σ[i]) < ε
            zeros_indices[count + 1] = i
            zeros_indices[count + 2] = i + 1
            maxval = max(maxval, Σ[i])
            count += 2
        else
            τ₀ = min(τ₀, abs(Σ[i] - maxval))
        end   
    end
    return zeros_indices[1:count], τ₀  #s₀ in the paper
end
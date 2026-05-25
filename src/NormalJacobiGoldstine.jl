using LinearAlgebra

"""
See

H. H. Goldstine and L. P. Horwitz. A procedure for the diagonalization of normal matrices. J. ACM, 6(2):176-195, 1959. https://doi.org/10.1145/320964.320975.

and

P. J. Eberlein. Solution to the complex eigenproblem by a norm reducing Jacobi type method. Numerische Mathematik, 14(3):232-245, 1970. https://doi.org/10.1007/BF02163332.
"""


"""
    offdiag(A::AbstractMatrix)

Compute the off-diagonal Frobenius norm of a symmetric matrix A.
"""
function offdiag(A::AbstractMatrix{<:Complex})
    Σ = 0
    n = size(A, 1)
    for i ∈ 1:n
        for j ∈ (i+1):n
            Σ += abs(A[i, j])^2 + abs(A[j, i])^2
        end
    end 
    return sqrt(Σ)
end

"""
    normal_jacobi_goldstine!(A::AbstractMatrix{<:Complex})
In-place Jacobi method for a normal matrix:\

    Goldstine, H. H., Horwitz, L. P.: A procedure for the diagonalization of normal matrices,
    J. ACM, 6(2), 176-195, (1959), https://doi.org/10.1145/320964.320975 
    
    Eberlein, P. J.: Solution to the complex eigenproblem by a norm reducing Jacobi type method,
    Numerische Mathematik, 14(3), 232-245, (1970), https://doi.org/10.1007/BF02163332
    
Input:  - a normal matrix `A`.\\
Output: The diagonal form of A in a `Diagonal` matrix.
"""
@views function normal_jacobi_goldstine!(A::AbstractMatrix{<:Complex})
    n = size(A, 1)
    T = eltype(A)
    εₘ = eps(real(T))
    ε = 10 * εₘ * norm(A)
    iter = 1
    itermax = 5 * n
    ii = zeros(Integer, 2)
    tv = zeros(T, n, 2)
    th = zeros(T, 2, n)
    oldoff = Inf
    offd = offdiag(A)
    while offd> ε && iter < itermax && offd < oldoff
        for i ∈ 1:n
            for j ∈ i+1:n
                if abs(A[i, j]) > εₘ || abs(A[j, i]) > εₘ
                    ii .= i, j
                    if abs(A[i, j] + A[j, i]')^2 + real(A[i, i] - A[j, j])^2 ≥ abs(A[i, j] - A[j, i]')^2 + imag(A[i, i] - A[j, j])^2
                        d  = real(A[i, j] + A[j, i])
                        α  = atan(imag(A[i, j] - A[j, i]),  d)
                        d₂ = real(A[i, i] - A[j, j])
                        x = 0.5 * atan(abs(A[i ,j] + A[j, i]') / d₂)
                    else
                        d  = imag(A[i, j]+ A[j, i])
                        α  = atan(- real(A[i, j] - A[j, i]),  d)
                        d₂ = imag(A[i, i] - A[j, j])
                        x = 0.5 * atan(abs(A[i ,j] - A[j, i]') / d₂)
                    end
                    s, c = sincos(x)
                    s *= exp(complex(0, -α))
                    tv .= A[:, ii]
                    @. A[:, i] =  c * tv[:, 1] + s * tv[:, 2]
                    @. A[:, j] = -s' * tv[:, 1] + c * tv[:, 2]
                    th .= A[ii, :]
                    @. A[i, :] =  c * th[1, :] + s' * th[2, :] 
                    @. A[j, :] = -s * th[1, :] + c * th[2, :] 
                end
            end
        end
        oldoff = offd
        offd = offdiag(A)
        iter += 1
    end

    if iter == itermax
        @warn "Maximum number of iterations reached in normal_jacobi_goldstine!"
    end

    return Diagonal(A)
end

normal_jacobi_goldstine(A::AbstractMatrix{<:Complex}) = normal_jacobi_goldstine!(copy(A))

@views function parallel_normal_jacobi_goldstine!(A::AbstractMatrix{<:Complex})
    n = size(A, 1)
    T = eltype(A)
    εₘ = eps(real(T))
    ε = 10 * εₘ * norm(A)
    iter = 1
    itermax = 5 * n
    ii = zeros(Integer, 2)
    tv = zeros(T, n, 2)
    th = zeros(T, 2, n)
    oldoff = Inf
    offd = offdiag(A)
    steps = parallel_cyclic_order(n)
    while offd> ε && iter < itermax && offd < oldoff
        for pairs ∈ steps
            Threads.@threads for pair in pairs
                i = minimum(pair)
                j = maximum(pair) 
                if abs(A[i, j]) > εₘ || abs(A[j, i]) > εₘ
                    ii .= i, j
                    if abs(A[i, j] + A[j, i]')^2 + real(A[i, i] - A[j, j])^2 ≥ abs(A[i, j] - A[j, i]')^2 + imag(A[i, i] - A[j, j])^2
                        d  = real(A[i, j] + A[j, i])
                        α  = atan(imag(A[i, j] - A[j, i]),  d)
                        d₂ = real(A[i, i] - A[j, j])
                        x = 0.5 * atan(abs(A[i ,j] + A[j, i]') / d₂)
                    else
                        d  = imag(A[i, j]+ A[j, i])
                        α  = atan(- real(A[i, j] - A[j, i]),  d)
                        d₂ = imag(A[i, i] - A[j, j])
                        x = 0.5 * atan(abs(A[i ,j] - A[j, i]') / d₂)
                    end
                    s, c = sincos(x)
                    s *= exp(complex(0, -α))
                    tv .= A[:, ii]
                    @. A[:, i] =  c * tv[:, 1] + s * tv[:, 2]
                    @. A[:, j] = -s' * tv[:, 1] + c * tv[:, 2]
                    th .= A[ii, :]
                    @. A[i, :] =  c * th[1, :] + s' * th[2, :] 
                    @. A[j, :] = -s * th[1, :] + c * th[2, :] 
                end
            end
        end
        oldoff = offd
        offd = offdiag(A)
        iter += 1
    end

    if iter == itermax
        @warn "Maximum number of iterations reached in normal_jacobi_goldstine!"
    end

    return Diagonal(A)
end

parallel_normal_jacobi_goldstine(A::AbstractMatrix{<:Complex}) = parallel_normal_jacobi_goldstine!(copy(A))
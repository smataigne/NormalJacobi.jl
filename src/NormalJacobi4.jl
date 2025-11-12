using LinearAlgebra


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
    return Σ
end
@views function normaljacobieberlein!(A::AbstractMatrix{<:Complex})
    n = size(A, 1)
    T = eltype(A)
    ε = 100 *eps(real(T)) * norm(A)
    iter = 1
    itermax = 5 * sqrt(n)
    ii = zeros(Integer, 2)
    tv  = zeros(T, n, 2)
    th= zeros(T, 2, n)
    while offdiag(A) > ε && iter < itermax
        #print("Accuracy at iter", iter, " : ", norm(A-Matrix(Tridiagonal(A))), "\n")
        for i ∈ 1:n
            for j ∈ i+1:n
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
        #display(offdiag(A))
        iter += 1
    end
    return Diagonal(A)
end

normaljacobieberlein(A::AbstractMatrix{<:Complex}) = normaljacobieberlein!(copy(A))

#n = 20
#A = Matrix(qr(randn(Float64, n , n)).Q)
#A = complex.(A, 0)
#display(eigvals(A))
#normaljacobieberlein!(A)
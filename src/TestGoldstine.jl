using LinearAlgebra

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

function myfindmax(A::AbstractMatrix{<:Complex})
    n = size(A, 1)
    maxval = 0.0
    maxi, maxj = 1, 1
    for i ∈ 1:n
        for j ∈ (i+1):n
            if i != j && abs(A[i, j]) > maxval
                maxval = abs(A[i, j])
                maxi, maxj = i, j
            end
        end
    end
    return maxi, maxj
end

@views function func(A::AbstractMatrix)
    n = size(A, 1)
    T = eltype(A)
    ε = 100 * eps(real(T)) * norm(A)
    δ = offdiag(A)
    iter = 0
    itermax = 10
    ii = zeros(Int, 2)
    temp1 = zeros(T, 2, n)
    temp2 = zeros(T, n, 2)
    while δ > ε && iter < itermax
        i, j = myfindmax(A)
        b = (A[i, j] + A[j, i]') / 2
        ω = (A[i, j] - A[j, i]') / 2 
        r = abs(b)
        β = angle(b) #atan(imag(b), real(b))
        s = abs(ω)
        γ = angle(ω) #atan(imag(ω), real(ω))
        ν = (iszero(r * s) ? π/2 : β - γ)
        u = abs(real(A[j, j])) - abs(real(A[i, i]))
        v = abs(imag(A[j, j])) - abs(imag(A[i, i]))
        t = complex(u, v)
        display(t)
        α = -0.1
        ϕ = 0.1
        κ = u^2 + v^2 - 4 * (r^2 * cos(β - α)^2 + s^2 * sin(γ - α)^2)
        λ = 4 * (r * u * cos(β - α) + s * v * sin(γ - α))
        #######
        q = complex(cos(α), sin(α))
        U = [cos(ϕ) (sin(ϕ) * q); (-q' * sin(ϕ)) cos(ϕ)]
        B =  copy(A)
        ii .= i, j
        oldδ = offdiag(A)
        temp1 .= A[ii, :]
        mul!(A[ii, :], U, temp1, 1, 0)
        temp2 .= A[:, ii]
        mul!(A[:, ii], temp2, U', 1, 0)
        #Check formulas on page 179
        #display(A[i,i])
        #Formula OK display(B[i,i] * cos(ϕ)^2 + B[j,j] * sin(ϕ)^2 + (B[i,j] * q' + B[j,i] * q) * sin(ϕ) * cos(ϕ))
        #display(A[j,j])
        #Formula OK display(B[j,j] * cos(ϕ)^2 + B[i,i] * sin(ϕ)^2 - (B[i,j] * q' + B[j,i] * q) * sin(ϕ) * cos(ϕ))
        #display(A[i,j])
        #Formula OK display(q * ((B[j, j] - B[i, i])/2 * sin(2ϕ) - B[j, i] * q * sin(ϕ)^2 + B[i, j] * q' * cos(ϕ)^2))
        #display(A[j, i])
        #Formula OK display(q' * ((B[j, j] - B[i, i])/2 * sin(2ϕ) - B[i, j] * q' * sin(ϕ)^2 + B[j, i] * q * cos(ϕ)^2))
        δ = offdiag(A)
        print("Iteration \n")
        display(δ - oldδ)
        display(0.5 * κ * sin(2ϕ)^2 + 0.25 * λ * sin(4ϕ))
        display(0.5 * κ * sin(2ϕ)^2 - 0.25 * λ * sin(4ϕ))
        iter += 1
    end  
end

n = 8
A = Matrix(qr(randn(ComplexF64, n , n)).Q)
func(A)
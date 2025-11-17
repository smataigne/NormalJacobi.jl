using LinearAlgebra

"""
    givens(a, b)
    
Compute the cosine and sine of the Givens rotation such that [c s; -s c] [a ;b] = [nm; 0.0].
"""
function givens(a, b)
    nm = hypot(a, b)
    return a/nm, b/nm  #c, s
end

"""
    paardekooper(a::Number, b::Number, e::Number, d::Number)
Paardekooper's method to compute the cosines and sines of the two Givens rotations such that\\
```
[c₁ s₁;     [a b;     [c₂ s₂;\
-s₁ c₁]'  * e d] *  -s₂ c₂] .
```
is diagonal.
"""
function paardekooper(a::Number, b::Number, e::Number, d::Number)
    #More stable implementation 
    #=
    ε = eps(typeof(a))
    ν₁ = a^2 + b^2 - d^2 - e^2
    if abs(ν₁) > ε
        κ =  2 * (a * e + b * d) / ν₁
        t₁ = κ /(1 + hypot(1, κ))               #tan(2α₁) = κ 
        c₁ = 1 / hypot(1, t₁)                   #cos(α₁) = 1/√(1+tan²(α₁))
        s₁ = c₁ * t₁                            #sin(α₁) = cos(α₁)*tan(α₁)
    else
        κ = a * e + b * d
        c₁ = 1 / √2 
        s₁ = sign(κ) * sign(ν₁)  *  c₁
    end
    ν₂ = c₁ * d - s₁ * b
    if abs(ν₂) > ε
        t₂ = (s₁ * a - c₁ * e) / ν₂
        c₂ = 1 / hypot(1, t₂)                   #cos(α₂) = 1/√(1+tan²(α₂))
        s₂ = c₂ * t₂                            #sin(α₂) = cos(α₂)*tan(α₂)
        return c₁, s₁, c₂, s₂
    else
        return c₁, s₁, 0, 1
    end
    =#
    #Unstable implementation
    α₁ = 0.5 * atan(2 * (a * e + b * d) / ( a^2 + b^2 - d^2 - e^2))
    c₁ = cos(α₁); s₁ = sin(α₁);
    α₂ = atan((s₁ * a - c₁ * e) / (c₁ * d - s₁ * b))
    c₂ = cos(α₂); s₂ = sin(α₂);
    return c₁, s₁, c₂, s₂
end

"""
    jacobi_sym(x11::Number, x12::Number, x22::Number)

Jacobi rotation for a symmetric 2x2 matrix [x11 x12; x12 x22].
"""
function jacobi_sym(x11::Number, x12::Number, x22::Number)
    T = typeof(x11)
    if iszero(x12)
        return T(1), T(0)
    end
    τ = (x11 - x22)/(2 * x12)
    t = sign(τ) /(abs(τ) + √(1 + τ * τ))
    c = 1 / √(1 + t * t)
    s = c * t
    return c, s
end


"""
    offSchur(A::AbstractMatrix)
Compute the off-Schur Frobenius norm of a matrix A.
"""
function offSchur(A::AbstractMatrix)
    Σ = 0
    n = size(A, 1)
    for i ∈ 1:2:n-2
        for j ∈ i+2:n
            Σ += A[j, i]^2 + A[j, i + 1]^2 
        end
    end
    return sqrt(Σ)
end

"""
    offdiag(A::AbstractMatrix)

Compute the off-diagonal Frobenius norm of a symmetric matrix A.
"""
function offdiag(A::AbstractMatrix{<:Real})
    Σ = 0
    n = size(A, 1)
    for i ∈ 1:n
        for j ∈ (i+1):n
            Σ += 2 * A[i, j]^2
        end
    end 
    return sqrt(Σ)
end

"""
    SSHjacobi!(A::AbstractMatrix{T}) where T

In-place Symmetric Skew-Hamiltonian Jacobi method for a symmetric matrix A of even size.
"""
@views function SSHjacobi!(A::AbstractMatrix{T}) where T
    n = size(A, 1)
    itermax = 10; iter = 1
    n2 = n ÷ 2
    ii = zeros(Int, 4)
    p = zeros(T, 3)
    R = zeros(T, 4, 4)
    temp1 = zeros(T, 4, n)
    temp2 = zeros(T, n, 4)
    ε = eps(T) * 100 * norm(A)
    while offdiag(A) > ε && iter < itermax
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
    end
    return A
end

"""
    findzeros!(Σ::AbstractVector{T}) where T
    
Find the indices of the (approximate) zeros in the vector Σ.
"""
function findzeros!(Σ::AbstractVector{T}) where T
    n = length(Σ)
    n1 = n + 1
    ε =  10 * eps(T) * norm(Σ)
    zeros_indices = zeros(Int, n)
    i = 1
    count = 1
    while i < n1 
        if Σ[i] < ε
            #push!(zeros_indices, i)
            zeros_indices[count] = i
            count += 1
            i += 1
        else
            i += 2
        end   
    end
    if i == n1 && Σ[end] < ε
        #push!(zeros_indices, n1)
        zeros_indices[count] = n1
        count += 1
    end
    count -= 1
    return zeros_indices[1:count]
end

"""
    create_matrix(n::Integer, α₁::Number, α₂::Number) -> Matrix{Float64}
Create an n x n real matrix with specified proportions of real and complex eigenvalues.
The parameters α₁ and α₂ determine the fractions of real and repeated complex eigenvalues, respectively
"""
function create_matrix(n::Integer, α₁::Number, α₂::Number)
    Q = Matrix(qr(randn(n, n)).Q)
    d = zeros(n)
    sd = zeros(n - 1) 
    a₁ = floor(Int, α₁ * n)
    if isodd(a₁)  && iseven(n)  # Ensure a₁ is even if n is even
        a₁ -= 1
    end
    a₂ = floor(Int, α₂ * n)
    if isodd(a₂)  
        a₂ -= 1
    end
    s = randn(a₂ ÷ 2)
    d[1:a₁] = randn(a₁)
    d[(a₁+1):(a₁+a₂)] = invpermute!([s; s], [1:2:a₂;2:2:a₂])
    sd[(a₁+1):2:(a₁+a₂ - 1)] .= randn()
    p = n - a₁ - a₂
    s = randn(p ÷ 2)
    d[(a₁ + a₂ + 1):end] = invpermute!([s; s], [1:2:p;2:2:p])
    sd[(a₁ + a₂ + 1):2:end] .= randn(p÷2)
    return Matrix(Q * Tridiagonal(sd, d, -sd) * Q')
end

"""
    create_matrix(n::Integer, α₁::Number, α₂::Number) -> Matrix{Float64}
Create an n x n real matrix with specified proportions of real and complex eigenvalues.
The parameters α₁ and α₂ determine the fractions of real and repeated complex eigenvalues, respectively
"""
function create_matrix2(n::Integer, α₁::Number, α₂::Number)
    Q = Matrix(qr(randn(n, n)).Q)
    d = zeros(n)
    sd = zeros(n - 1) 
    a₁ = floor(Int, α₁ * n)
    if isodd(a₁)  && iseven(n)  # Ensure a₁ is even if n is even
        a₁ -= 1
    end
    a₂ = floor(Int, α₂ * n)
    if isodd(a₂)  
        a₂ -= 1
    end
    s = randn(a₂ ÷ 2)
    d[1:a₁] = randn(a₁)
    d[(a₁+1):(a₁+a₂)] = invpermute!([s; s], [1:2:a₂;2:2:a₂])
    sd[(a₁+1):2:(a₁+a₂ - 1)] .= randn()
    p = n - a₁ - a₂
    s = randn(p ÷ 2)
    d[(a₁ + a₂ + 1):end] = invpermute!([s; s], [1:2:p;2:2:p])
    sd[(a₁ + a₂ + 1):2:end] .= 1 .+  0.01*√(eps(Float64))  * randn(p÷2)
    return Matrix(Q * Tridiagonal(sd, d, -sd) * Q')
end

@views function create_matrix3(θs::AbstractVector, λs::AbstractVector, κ::AbstractVector)
    p = length(θs) 
    r = length(λs)
    n = 2p + r
    QR = qr(randn(n, n))
    Q = Matrix(QR.Q) * Diagonal(sign.(diag(QR.R)))
    M = similar(Q, n, n)
    for (i, θ) ∈  enumerate(θs)
        c = κ[i] * cos(θ); s = κ[i] * sin(θ)
        j = 2i - 1
        M[:, j]     .=  c * Q[:, j] + s * Q[:, j + 1]
        M[:, j + 1] .= -s * Q[:, j] + c * Q[:, j + 1]
    end
    for (i, λ) ∈ enumerate(λs)
        M[:, 2p + i] .= (λ .* Q[:, 2p+i])
    end
    return M * Q'
end
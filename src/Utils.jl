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
    #Simple implementation
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
function offdiag(A::AbstractMatrix)
    Σ = 0
    n = size(A, 1)
    for i ∈ 1:n
        for j ∈ (i+1):n
            Σ += 2 * abs(A[i, j])^2
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


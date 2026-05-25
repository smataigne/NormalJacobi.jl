using LinearAlgebra

"""
    issym(A::AbstractMatrix{T}, ε::Number) where T
    
Check if the matrix A is symmetric within a tolerance ε.
"""
function is_sym(A::AbstractMatrix{T}, ε::Number) where T
    n = size(A, 1)
    for i ∈ 1:n
        for j ∈ i+1:n
            if abs(A[i, j] - A[j, i]') > ε
                return false
            end
        end
    end
    return true
end

"""
    ssh_offdiag(A::AbstractMatrix{T}) where T

Compute the off-diagonal Frobenius norm of a symmetric skew-Hamiltonian part of a matrix A.
"""
@views function ssh_offdiag(A::AbstractMatrix{T}) where T
    n = size(A, 1)
    m = n ÷ 2
    shuffle = [1:2:n-1; 2:2:n]
    temp = zeros(T, n, n)
    temp .= A[shuffle, shuffle]
    M1 = temp[1:m, 1:m] 
    M1 .+= temp[m+1:end, m+1:end]
    M1 .+= M1'
    M1 .*= 0.25
    for i ∈ 1:m
        M1[i, i] = 0
    end
    M2 = temp[m+1:end, 1:m] 
    M2 .-= temp[1:m, m+1:end]
    M2 .-= M2'
    M2 .*= 0.25
    return hypot(sqrt(2) * norm(M1), sqrt(2) * norm(M2))
end

"""
    skew_offschur(A::AbstractMatrix{T}) where T

Compute the norm of the skew-symmetric part.
"""
function skew_offschur(A::AbstractMatrix{T}) where T
    Σ = zero(T)
    n = size(A, 1)
    for i ∈ 1:2:n-3
        for j ∈ i+2:2:n-1
            Σ = hypot(Σ, ((A[j, i] - A[i, j]') / sqrt(2)))
            Σ = hypot(Σ, ((A[j + 1, i] - A[i, j + 1]') / sqrt(2)))
            Σ = hypot(Σ, ((A[j, i + 1] - A[i + 1, j]') / sqrt(2)))
            Σ = hypot(Σ, ((A[j + 1, i + 1] - A[i + 1, j + 1]') / sqrt(2)))
        end
    end
    return Σ
end

"""
    sym_offdiag(A::AbstractMatrix{T}) where T

Compute the norm of the symmetric part.
"""
function sym_offdiag(A::AbstractMatrix{T}) where T
    Σ = zero(T)
    n = size(A, 1)
    for i ∈ 1:n
        for j ∈ i+1:n
            Σ = hypot(Σ, ((A[j, i] + A[i, j]') / sqrt(2)))
        end
    end
    return Σ
end

"""
    ssh_part(A::AbstractMatrix{T}) where T

Returns the symmetric skew-Hamiltonian part of a matrix (shuffled).
"""
@views function ssh_part(A::AbstractMatrix{T}) where T
    n = size(A, 1)
    m = n ÷ 2
    shuffle = [1:2:n-1; 2:2:n]
    temp = zeros(T, n, n)
    temp .= A[shuffle, shuffle]
    M1 = temp[1:m, 1:m] 
    M1 .+= temp[m+1:end, m+1:end]
    M1 .+= M1'
    M1 .*= 0.25
    M2 = temp[m+1:end, 1:m] 
    M2 .-= temp[1:m, m+1:end]
    M2 .-= M2'
    M2 .*= 0.25
    temp[m+1:end, m+1:end] .= M1
    temp[1:m, m+1:end] .= -M2
    kk = invpermute!(Array(1:n), [1:2:n-1; 2:2:n])
    return temp[kk, kk]
end

function is_ssh(A::AbstractMatrix{T}, ε::Number) where T
    n = size(A, 1)
    W₁ = zeros(T, 2, 2)
    X₁ = zeros(T, 2, 2)
    W₂ = zeros(T, 2, 2)
    X₂ = zeros(T, 2, 2)
    W = zeros(T, 2, 2)
    X = zeros(T, 2, 2)
    σ = 0
    for i ∈ 1:2:n-1
        σ += A[i + 1, i]
    end
    σ *= 2 / n
    for i ∈ 1:2:n-3
        for j ∈ i+2:2:n-1
            W₁[1, 1] = A[i, i]
            W₁[1, 2] = A[i, j]
            W₁[2, 1] = A[j, i]
            W₁[2, 2] = A[j, j]
            W₂[1, 1] = A[i + 1, i + 1]
            W₂[1, 2] = A[i + 1, j + 1]
            W₂[2, 1] = A[j + 1, i + 1]
            W₂[2, 2] = A[j + 1, j + 1]
            X₁[1, 1] = A[i + 1, i] - σ
            X₁[1, 2] = A[i + 1, j]
            X₁[2, 1] = A[j + 1, i]
            X₁[2, 2] = A[j + 1, j] - σ
            X₂[1, 1] = A[i, i + 1] + σ
            X₂[1, 2] = A[i, j + 1]
            X₂[2, 1] = A[j, i + 1]
            X₂[2, 2] = A[j, j + 1] + σ
            W = (W₁ + W₂) / 2
            W .+= W'
            W .*= 0.5
            X = (X₁ - X₂) / 2
            X .-= X'
            X .*= 0.5
            e = sqrt(norm(W₁ - W)^2 + norm(W₂ - W)^2 + norm(X₁ - X)^2 + norm(X₂ + X)^2)
            if e > ε
                return false
            end
        end
    end
    return true
end

"""
    givens(a, b)
    
Compute the cosine and sine of the Givens rotation such that [c s; -s c] [a ;b] = [nm; 0.0].
"""
function givens(a, b)
    nm = hypot(a, b)
    return a/nm, b/nm  #c, s
end

"""
    annihilator(a::Number, b::Number, e::Number, d::Number)
Paardekooper's method to compute the cosines and sines of the two Givens rotations such that\\
```
[c₁ s₁;     [a b;     [c₂ s₂;\
-s₁ c₁]'  * e d] *  -s₂ c₂] .
```
is diagonal.
"""
function annihilator(a::Number, b::Number, e::Number, d::Number)
    #=
    #More stable implementation 
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
    #More stable implementation
    α₁ = 0.5 * atan(2 * (a * e + b * d), ( a^2 + b^2 - d^2 - e^2))
    s₁, c₁ = sincos(α₁);
    α₂ = atan((s₁ * a - c₁ * e), (c₁ * d - s₁ * b))
    s₂, c₂ = sincos(α₂);
    return c₁, s₁, c₂, s₂
end

"""
    jacobi_rotation(x11::Number, x12::Number, x22::Number)

Jacobi rotation for a symmetric 2x2 matrix [x11 x12; x12 x22].
"""
function jacobi_rotation(x11::Number, x12::Number, x22::Number)
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
    offschur(A::AbstractMatrix)
Compute the off-Schur Frobenius norm of a matrix A.
"""
function offschur(A::AbstractMatrix)
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
    offschurl(A::AbstractMatrix)
Compute the off-Schur Frobenius norm of the lower triangle of the matrix A.
"""
function offSchurl(A::AbstractMatrix)
    Σ = 0
    n = size(A, 1)
    for i ∈ 1:2:n-2
        for j ∈ i+2:n
            Σ += A[j, i]^2
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
    build_adjacency(A::AbstractMatrix{T}, ε::Number) where T

Build the adjacency matrix for the graph of the normal matrix `A`, where two vertices i and j are connected if the 2x2 block A[j:j+1, i:i+1] has norm greater than ε.
"""
function build_adjacency(A::AbstractMatrix{T}, ε::Number) where T
    n = size(A, 1)
    adj = zeros(Bool, n, n)
    for i ∈ 1:2:n-3
        adj[i:i+1, i:i+1] .= true
        for j ∈ i+2:2:n-1
            if norm(A[j:j+1, i:i+1], Inf) > ε
                adj[j:j+1, i:i+1] .= true
                adj[i:i+1, j:j+1] .= true
            end
        end
    end
    return adj
end

"""
    find_connected_components(adj::AbstractMatrix{Bool})
"""
function find_connected_components(adj::AbstractMatrix{Bool})
    n = size(adj, 1)
    visited = zeros(Bool,n)
    components = Vector{Vector{Int}}()
    for i ∈ 1:n
        if !visited[i]
            component = Int[] #Initialize an empty component
            tovisit = [i]     #Use a stack for depth-first search
            while !isempty(tovisit)
                node = pop!(tovisit)
                if !visited[node]
                    push!(component, node)
                    visited[node] = true
                    for j ∈ 1:n
                        if adj[node, j] && !visited[j]
                            push!(tovisit, j)
                        end
                    end
                end
            end
            if length(component) > 2
                push!(components, sort!(component))
            end
        end
    end
    return components
end

@views function parallel_cyclic_order(n)

    idx = collect(1:n)
    steps = zeros(Int, n - 1, n ÷ 2, 2)
    for j in 1:n-1
        for i in 1:(n ÷ 2)
            steps[j, i, :] .= idx[i], idx[n-i+1]
        end
        last = idx[end]
        for i in n:-1:3
            idx[i] = idx[i-1]
        end
        idx[2] = last
    end

    return steps
end


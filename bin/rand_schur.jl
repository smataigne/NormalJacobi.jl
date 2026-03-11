using LinearAlgebra
include("../src/Utils.jl")
include("../src/NormalJacobi.jl")
include("../src/NormalJacobiBunse.jl")
include("../src/NormalJacobiGoldstine.jl")

@views function randschur!(A::AbstractMatrix)
    n = size(A, 1)
    T = typeof(A[1, 1])
    εₘ = eps(T)
    cc = maximum(abs.(A))
    ε  = 10 * εₘ * norm(A)
    iter = 1
    itermax = 5 * n
    indices = zeros(Integer, 4)
    temp1  = zeros(T, n, 4)
    temp2 = zeros(T, 4, n)
    oldoff = Inf
    offschur = offSchurl(A)
    while iter < itermax && offschur > ε
        for i ∈ 1:2:n-2
            for j ∈ i+2:2:n-1
                indices .= i, i+1, j, j+1
                if norm(A[[j, j + 1],[i, i + 1]]) > 4εₘ * cc
                    _, Q, v = schur(A[indices, indices])
                    if iszero(imag(v[1])) && !iszero(imag(v[2])) 
                        Base.permutecols!!(Q, [2, 3, 1, 4])
                    end
                
                    mul!(temp2, Q' , A[indices, :], 1, 0)
                    A[indices, :] .= temp2
                    mul!(temp1, A[:, indices], Q, 1, 0)
                    A[:, indices] .= temp1
                end
            end
        end
        oldoff = offschur
        offschur = offSchurl(A)
        iter += 1
        #display(A)
        display(offschur)
    end
    
    if iter == itermax
        @warn "Maximum number of iterations reached in randschur!"
    end

    return A
end

@views function randschur2!(A::AbstractMatrix)
    n = size(A, 1)
    T = typeof(A[1, 1])
    εₘ = eps(T)
    cc = maximum(abs.(A))
    ε  = 10 * εₘ * norm(A)
    iter = 1
    itermax = 5 * n
    indices = zeros(Integer, 2)
    temp1  = zeros(T, n, 2)
    temp2 = zeros(T, 2, n)
    oldoff = Inf
    offschur = offSchurl(A)
    while iter < itermax && offschur > ε
        for i ∈ 1:2:n-1
            for j ∈ i+1:n
                p = j#findmax(abs.(A[i+1:end, i]))[2] + i
                indices .= i, p
                if norm(A[j, i]) > εₘ * cc
                    #=
                    R = hypot(A[i, i] - A[j, j], A[i, j] + A[j, i])
                    ϕ = atan(A[i, i] - A[j, j], A[i, j] + A[j, i])
                    t = 0.5 * (acos((A[i, j] - A[j, i]) / R) + ϕ)
                    Q = [cos(t) sin(t); -sin(t) cos(t)]
                    =#
                    _, Q, _ = schur(A[indices, indices])
                    mul!(temp2, Q' , A[indices, :], 1, 0)
                    A[indices, :] .= temp2
                    mul!(temp1, A[:, indices], Q, 1, 0)
                    A[:, indices] .= temp1
                end
            end
        end
        oldoff = offschur
        offschur = offSchurl(A)
        iter += 1
        #display(A)
        display(offschur)
    end
    
    if iter == itermax
        @warn "Maximum number of iterations reached in randschur!"
    end

    return A
end
#=
A = rand(2, 2)
display(A)
R = hypot(A[1,1] - A[2, 2], A[1, 2] + A[2, 1])
ϕ = atan(A[1, 1] - A[2, 2], A[1, 2] + A[2, 1])
t = 0.5 * (acos((A[1,2] - A[2, 1]) / R) + ϕ)
Q = [cos(t) sin(t); -sin(t) cos(t)]
display(Q'A*Q)
display(schur(A).T)
=#
A = randn(10, 10)
randschur!(A)

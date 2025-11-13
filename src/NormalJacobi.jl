using LinearAlgebra, SkewLinearAlgebra, LaTeXStrings
include("Utils.jl")
include("NormalJacobiZhou.jl")
"""
    offdiag(A::AbstractMatrix)

Compute the off-diagonal Frobenius norm of a symmetric matrix A.
"""
function offdiag(A::AbstractMatrix)
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

@views function normal_skew_jacobi!(A::AbstractMatrix{T}, showphase::Bool) where T
    n = size(A, 1)
    ε = eps(T) * norm(A) * sqrt(n)   #Matrix-wise norm bound
    εₘ = eps(T) * 10                 #Element-wise norm bound
    μ = sqrt(eps(T)) * norm(A)
    ii = zeros(Int64, 2)             #Indices for rows/columns selections
    th = zeros(T, 2, n)
    tv = zeros(T, n, 2)
    iter = 1; itermax = 5 * sqrt(n)
    Ω = zeros(n, n)
    Ω .= (A .- A') / 2
    offschur = offSchur(Ω)
    if showphase == true
        P = plot(framestyle=:none, legend=:topright,font="Computer Modern", tickfontfamily="Computer Modern",legendfont="Computer Modern", guidefontfamily = "Computer Modern",
legendfontsize=10,yguidefontsize=13,xguidefontsize=13, xtickfontsize = 13, ytickfontsize=13, margin = 0.3Plots.cm, minorgrid = false, titlefontfamily="Computer Modern", titlefontsize=20, aspect_ratio=1)
    heatmap!(log10.(max.(abs.(A), eps(Float64))), colormap=:viridis, xticks=false, yticks=false, colorbar_fontsize=1, clim=(-15, 0), colorbar=false; yflip=true)
        title!(L"Phase 0: Initial Matrix $A$")
        savefig(P, "NormalJacobi_phase0.pdf")
    end
    #Phase 1
    while offschur > ε && iter < itermax
        for i ∈ 1:2:n-3
            for j ∈ i+2:2:n-1
                #First Jacobi Annihilator
                #c₁, s₁, c₂, s₂ = paardekooper(Ω[i + 1, i], Ω[i + 1, j], Ω[j + 1, i], Ω[j + 1, j])
                ω1 = (A[i + 1, i] - A[i, i + 1]) / 2
                ω2 = (A[i + 1, j] - A[j, i + 1]) / 2
                ω3 = (A[j + 1, i] - A[i, j + 1]) / 2
                ω4 = (A[j + 1, j] - A[j, j + 1]) / 2
                if abs(ω2) + abs(ω3) > εₘ
                    c₁, s₁, c₂, s₂ = paardekooper(ω1, ω2, ω3, ω4)
                    #First Similarity transformation "G1' * A * G1"
                    #G1 = [c₁ -s₁; s₁ c₁]
                    #A[[i+1, j+1], :] = G1'A[[i+1, j+1], :]
                    #A[:, [i+1, j+1]] = A[:, [i+1, j+1]] * G1
                    ii .= i + 1, j + 1
                    th .= A[ii, :]
                    @. A[i + 1, :] =  c₁ * th[1, :] + s₁ * th[2, :] 
                    @. A[j + 1, :] = -s₁ * th[1, :] + c₁ * th[2, :] 
                    tv .= A[:, ii]
                    @. A[:, i + 1] =  c₁ * tv[:, 1] + s₁ * tv[:, 2]
                    @. A[:, j + 1] = -s₁ * tv[:, 1] + c₁ * tv[:, 2]

                    #Second Similarity transformation "G2' * A * G2"
                    #G2 = [c₂ -s₂; s₂ c₂]
                    #A[:, [i, j]] = A[:, [i, j]] * G2
                    #A[[i, j], :] = G2'A[[i, j], :]
                    ii .= i, j
                    tv .= A[:, ii]
                    @. A[:, i] =  c₂ * tv[:, 1] + s₂ * tv[:, 2]
                    @. A[:, j] = -s₂ * tv[:, 1] + c₂ * tv[:, 2]
                    th .= A[ii, :]
                    @. A[i, :] =  c₂ * th[1, :] + s₂ * th[2, :] 
                    @. A[j, :] = -s₂ * th[1, :] + c₂ * th[2, :] 
                end
                #c₁, s₁, c₂, s₂ = paardekooper(Ω[i + 1, i], Ω[i + 1, j + 1], Ω[j, i], Ω[j, j + 1])
                ω1 = (A[i + 1, i] - A[i, i + 1]) / 2
                ω2 = (A[i + 1, j + 1] - A[j + 1, i + 1]) / 2
                ω3 = (A[j, i] - A[i, j]) / 2
                ω4 = (A[j, j + 1] - A[j + 1, j]) / 2
                if abs(ω2) + abs(ω3) > εₘ     
                    c₁, s₁, c₂, s₂ = paardekooper(ω1, ω2, ω3, ω4)
                    #First Similarity transformation "G1' * A * G1"
                    #G1 = [c₁ -s₁; s₁ c₁]
                    ii .= i + 1, j
                    th .= A[ii, :]
                    @. A[i + 1, :] =  c₁ * th[1, :] + s₁ * th[2, :] 
                    @. A[j, :] = -s₁ * th[1, :] + c₁ * th[2, :] 
                    tv .= A[:, ii]
                    @. A[:, i + 1] =  c₁ * tv[:, 1] + s₁ * tv[:, 2]
                    @. A[:, j] = -s₁ * tv[:, 1] + c₁ * tv[:, 2]
                    #Second Similarity transformation "G2' * A * G2"
                    #G2 = [c₂ -s₂; s₂ c₂]
                    ii .= i, j + 1
                    tv .= A[:, ii]
                    @. A[:, i] =  c₂ * tv[:, 1] + s₂ * tv[:, 2]
                    @. A[:, j + 1] = -s₂ * tv[:, 1] + c₂ * tv[:, 2]
                    th .= A[ii, :]
                    @. A[i, :] =  c₂ * th[1, :] + s₂ * th[2, :] 
                    @. A[j + 1, :] = -s₂ * th[1, :] + c₂ * th[2, :]
                end
            end
        end
        Ω .= (A .- A') / 2
        offschur = offSchur(Ω)
        iter +=1
    end
    if showphase == true
        P = plot(framestyle=:none, legend=:topright,font="Computer Modern", tickfontfamily="Computer Modern",legendfont="Computer Modern", guidefontfamily = "Computer Modern",
legendfontsize=10,yguidefontsize=13,xguidefontsize=13, xtickfontsize = 13, ytickfontsize=13, margin = 0.3Plots.cm, minorgrid = false, titlefontfamily="Computer Modern", titlefontsize=20, aspect_ratio=1)
    heatmap!(log10.(max.(abs.(A), eps(Float64))), colormap=:viridis, xticks=false, yticks=false, colorbar_fontsize=1, clim=(-15, 0), colorbar=false; yflip=true)
        title!(L"Phase 1: $A$ after Paardekooper's Method")
        savefig(P, "NormalJacobi_phase1.pdf")
    end
    if offSchur(Ω) > ε
        @warn "Maximum number of iterations reached in Phase 1 with offSchur = $(offSchur(Ω))"
    end
    if offSchur(A) < ε
        return Tridiagonal(A) 
    end
    Σ = zeros(T, n-1)
    for i ∈ 1:n-1
        Σ[i] = abs(Ω[i+1, i])
    end
    kk = findzeros!(Σ)
    #Phase 2: Symmetric Jacobi
    M = A[kk,kk]
    M .+= M'
    M .*= 0.5
    K = length(kk)
    th = zeros(T, 2, K)
    tv = zeros(T, K, 2)
    iter = 1; itermax = max(5 * sqrt(K), 15)
    while offdiag(M) > ε && iter < itermax
        for i ∈ 1:K-1
            for j ∈ i+1:K
                if abs(M[j, i]) > ε / sqrt(n)
                    c, s = jacobi_sym(M[i, i], M[j, i], M[j, j])
                    #G = [c -s; s c]
                    #M[[i, j], :] = G' * M[[i, j], :]
                    #M[:, [i, j]] = M[:, [i, j]] * G
                    ii .= i, j
                    th .= M[ii, :]
                    @. M[i, :] =  c * th[1, :] + s * th[2, :] 
                    @. M[j, :] = -s * th[1, :] + c * th[2, :] 
                    tv .= M[:, ii]
                    @. M[:, i] =  c * tv[:, 1] + s * tv[:, 2]
                    @. M[:, j] = -s * tv[:, 1] + c * tv[:, 2]
                end
            end
        end
        iter += 1
    end
    if showphase == true
        P = plot(framestyle=:none, legend=:topright,font="Computer Modern", tickfontfamily="Computer Modern",legendfont="Computer Modern", guidefontfamily = "Computer Modern",
legendfontsize=10,yguidefontsize=13,xguidefontsize=13, xtickfontsize = 13, ytickfontsize=13, margin = 0.3Plots.cm, minorgrid = false, titlefontfamily="Computer Modern", titlefontsize=20, aspect_ratio=1)
    heatmap!(log10.(max.(abs.(A), eps(Float64))), colormap=:viridis, xticks=false, yticks=false, colorbar_fontsize=1, clim=(-15, 0), colorbar=false; yflip=true)
        title!("Phase 2: After Symmetric Jacobi")  
    savefig(P, "NormalJacobi_phase2.pdf")
    end
    if offdiag(M) > ε
        @warn "Maximum number of iterations reached in Phase 2 with offdiag = $(offdiag(M))"
    end
    #Phase 3: Symmetric Skew-Hamiltonian Jacobi algorithm
    solved = zeros(Bool, n)
    for i ∈ 1:n-1
        if (A[i + 1, i] - A[i, i + 1]) < 0
            A[:, i + 1] .*= -1
            A[i + 1, :] .*= -1
        end
    end
    Ω .= (A .- A') / 2
    #display(A)
    i = 1
    l = zeros(Int, n)
    while i < n
        if abs(Σ[i]) > ε && !solved[i]
            #kk = [i, i+1]
            l[1] = i  
            l[2] = i + 1
            nk = 2     
            solved[i:i+1] .= true
            for j ∈ i+1:n-1
                if abs(Σ[j] - Σ[i]) < ε && !solved[j]
                    #push!(kk, j, j + 1)
                    nk += 2
                    l[nk - 1] = j  
                    l[nk] = j + 1
                    solved[j:j+1] .= true
                end
            end
            #nk = length(kk)
            kk = l[1:nk]
            if nk > 2
                #display(kk)

                M = copy(A[kk, kk] + A[kk, kk]')
                M .*= 0.5
                indices = [1:2:nk; 2:2:nk]
                SSHjacobi!(M[indices, indices])
                @. A[kk,kk] = M + Ω[kk, kk] 
            else
                solved[i:i+1] .= false
            end
        end
        i += 1
    end
    if showphase == true
        P = plot(framestyle=:none, legend=:topright,font="Computer Modern", tickfontfamily="Computer Modern",legendfont="Computer Modern", guidefontfamily = "Computer Modern",
legendfontsize=10,yguidefontsize=13,xguidefontsize=13, xtickfontsize = 13, ytickfontsize=13, margin = 0.3Plots.cm, minorgrid = false, titlefontfamily="Computer Modern", titlefontsize=20, aspect_ratio=1)
    heatmap!(log10.(max.(abs.(A), eps(Float64))), colormap=:viridis, xticks=false, yticks=false, colorbar_fontsize=1, clim=(-15, 0), colorbar=false; yflip=true)
        title!("Phase 3: After Skew-Hamiltonian Jacobi")    
    savefig(P, "NormalJacobi_phase3.pdf")
    end
    
    #Phase 4: Accuracy loss correction
    i = 1
    while i < n
        if abs(Σ[i]) > ε && !solved[i]
            l[1] = i  
            l[2] = i + 1
            nk = 2  
            solved[i:i+1] .= true
            for j ∈ i+1:n-1
                if abs(Σ[j] - Σ[i]) < μ && !solved[j]
                    #push!(kk, j, j + 1)
                    nk += 2
                    l[nk - 1] = j  
                    l[nk] = j + 1
                    solved[j:j+1] .= true
                end
            end
            #nk = length(kk)
            kk = l[1:nk]
            if nk > 2
                normal_jacobi_zhou!(A[kk, kk])
            else
                solved[i:i+1] .= false
            end
        end
        i += 1
    end
    if showphase == true
       P = plot(framestyle=:none, legend=:topright,font="Computer Modern", tickfontfamily="Computer Modern",legendfont="Computer Modern", guidefontfamily = "Computer Modern",
legendfontsize=10,yguidefontsize=13,xguidefontsize=13, xtickfontsize = 13, ytickfontsize=13, margin = 0.3Plots.cm, minorgrid = false, titlefontfamily="Computer Modern", titlefontsize=20, aspect_ratio=1)
heatmap!(log10.(max.(abs.(A), eps(Float64))), colormap=:viridis, xticks=false, yticks=false, colorbar_fontsize=1, clim=(-15, 0), colorbar=false; yflip=true)
        title!("Phase 4: After Accuracy Improvement")
        savefig(P, "NormalJacobi_phase4.pdf")
    end
    
    return Tridiagonal(A)
end

normal_skew_jacobi(A::AbstractMatrix{T}) where T = normal_skew_jacobi!(copy(A), false)







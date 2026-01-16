using LinearAlgebra, SkewLinearAlgebra, LaTeXStrings, PyPlot, PyCall
include("Utils.jl")
include("NormalJacobiZhou.jl")
include("NormalJacobiBunse.jl")
include("UtilsJacobi.jl")


@views function normal_skew_jacobi_plots!(A::AbstractMatrix{T}, showphase::Bool) where T
    PyPlot.matplotlib.rc("text", usetex=true)
    PyPlot.matplotlib.rc("font", family="serif", serif=["Computer Modern"])
    PyPlot.matplotlib.rc("mathtext", fontset="cm")
    pnorm = PyCall.pyimport("matplotlib.colors").LogNorm(
    vmin = eps(Float64),
    vmax = 1
    )
    n = size(A, 1)
    ε  = eps(T) * norm(A) * 10        #Matrix-wise norm bound
    n2 = opnorm(A) * 10
    εₘ = eps(T)                  #Element-wise norm bound
    η = 10 * n / n2
    μ  = η * ε                  #Target accuracy for clustering
    μₘ = η * εₘ                #Element-wise target accuracy for clustering
    ii = zeros(Int64, 2)         #Indices for rows/columns selections
    th = zeros(T, 2, n)
    tv = zeros(T, n, 2)
    iter = 1; itermax = 5 * sqrt(n)
    sep = 0.0
    Ω = zeros(n, n)
    Ω .= (A .- A') / 2
    oldoff = Inf
    offschur = offSchur(Ω)
    if showphase == true
        fig, ax = subplots()
        c = ax.imshow(max.(abs.(A), eps(Float64)), cmap="viridis", aspect="equal", norm=pnorm)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Initial Matrix " * L"A", fontsize =24)
        fig.savefig("./figures/NormalJacobi_phase0.pdf")
    end
    #Phase I  Implicit Paardekooper
    while offschur > ε && iter < itermax && offschur < oldoff
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
        oldoff = offschur
        offschur = offSchur(Ω)
        iter +=1
    end

    if showphase == true
        fig, ax = subplots()
        c = ax.imshow(max.(abs.(A), eps(Float64)), cmap="viridis", aspect="equal", norm=pnorm)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("After Phase I", fontsize =24)
        fig.savefig("./figures/NormalJacobi_phase1.pdf")
    end
    #Early stopping
    if offSchur(A) < ε
        return Tridiagonal(A) 
    end

    Σ = zeros(T, n-1)   #Singular values on the subdiagonal of Ω 
    for i ∈ 1:n-1
        Σ[i] = abs(Ω[i+1, i])
    end
    kk, τ₀ = findzeros!(Σ, εₘ * n2)
    solved = zeros(Bool, n)
    tempsolved = zeros(Bool, n)

    #Phase II: Implicit Symmetric Jacobi

    #Case II.1: Well separated clusters
    if length(kk) > 0
        if εₘ < μₘ * τ₀
            
            solved[kk] .= true
            sep = εₘ * n2     #Remember separation threshold
            K = length(kk)
            iter = 1; itermax = max(5 * sqrt(K), 15)
            oldoff = Inf
            M = zeros(T, K, K)
            M .= A[kk, kk] 
            M .+= A[kk, kk]'
            M .*= 0.5
            offdiagM = offdiag(M)
            while offdiagM > ε && iter < itermax && offdiagM < oldoff
                for i ∈ 1:K-1
                    for j ∈ i+1:K
                        r = (A[kk[j], kk[i]] + A[kk[j], kk[i]]) / 2
                        if abs(r) > εₘ
                            c, s = jacobi_sym(A[kk[i], kk[i]], r, A[kk[j], kk[j]])
                            ii .= kk[i], kk[j]
                            th .= A[ii, :]
                            @. A[ii[1], :] =  c * th[1, :] + s * th[2, :] 
                            @. A[ii[2], :] = -s * th[1, :] + c * th[2, :] 
                            tv .= A[:, ii]
                            @. A[:, ii[1]] =  c * tv[:, 1] + s * tv[:, 2]
                            @. A[:, ii[2]] = -s * tv[:, 1] + c * tv[:, 2]
                        end
                    end
                end
                iter += 1
                oldoff = offdiagM
                M .= A[kk, kk] 
                M .+= A[kk, kk]'
                M .*= 0.5
                offdiagM = offdiag(M)
            end
            #println("Done Phase 2 - Case II.1, nk = ", length(kk))
        else
            #Case II.2: Clustered singular values
            γ = εₘ * n2
            while εₘ > μₘ * τ₀ && length(kk) < n
                γ += (τ₀ + εₘ)  #Increase clustering threshold
                kk, τ₀ = findzeros!(Σ, γ)
            end
            solved[kk] .= true
            sep = γ  #Remember separation threshold
            normal_jacobi_bunse2!(A, kk)
            #println("Done Phase 2 - Case II.2,  nk = ", length(kk))
        end
    end 
    if showphase == true
        fig, ax = subplots()
        c = ax.imshow(max.(abs.(A), eps(Float64)), cmap="viridis", aspect="equal", norm=pnorm)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("After Phase II", fontsize =24)
        fig.savefig("./figures/NormalJacobi_phase2.pdf")
    end

    #Phase 3: Implicit Symmetric Skew-Hamiltonian Jacobi algorithm
    
    #Ensures correct signs on Ω
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
        if !solved[i]
            tempsolved .= solved
            l[1] = i  
            l[2] = i + 1
            nk = 2     
            solved[i:i+1] .= true
            τ = Inf
            for j ∈ i+2:2:n-1
                if abs(Σ[j] - Σ[i]) < (εₘ * n2) && !solved[j]
                    nk += 2
                    l[nk - 1] = j  
                    l[nk] = j + 1
                    solved[j:j+1] .= true
                else
                    τ = min(τ, abs(Σ[j] - Σ[i]))
                end
            end
            if nk > 2
                if εₘ < μₘ * τ
                    #Case III.1: Well separated clusters
                    kk = l[1:nk]
                    SSHjacobi2!(A, kk)
                    tempsolved .= solved
                    #println("Done Phase 3 - Case III.1,  nk = ", length(kk))
                else
                    #Case III.2: Clustered singular values
                    γ = εₘ * n2
                    nk = 1; nkold = 0
                    #Recompute clusters until well separated
                    while εₘ > μₘ * τ && nkold < nk
                        nkold = nk
                        solved .= tempsolved
                        γ += (τ + εₘ)  #Increase clustering threshold
                        l[1] = i  
                        l[2] = i + 1
                        nk = 2     
                        solved[i:i+1] .= true
                        τ = Inf
                        for j ∈ i+1:n-1
                            if abs(Σ[j] - Σ[i]) < γ && !solved[j]
                                #push!(kk, j, j + 1)
                                nk += 2
                                l[nk - 1] = j  
                                l[nk] = j + 1
                                solved[j:j+1] .= true
                            else
                                τ = min(τ, abs(Σ[j] - Σ[i]))
                            end
                        end
                    end
                    kk = copy(l[1:nk])
                    normal_jacobi_bunse2!(A, kk)
                    tempsolved .= solved
                    #println("Done Phase 3 - Case III.2, nk = ", length(kk))
                end
            end
        end
        i += 2
    end
    if showphase == true
        fig, ax = subplots()
        c = ax.imshow(max.(abs.(A), eps(Float64)), cmap="viridis", aspect="equal", norm=pnorm)
        cb = fig.colorbar(c, ax=ax, norm=norm)
        ticks = [eps(Float64), sqrt(eps(Float64)), 1.0]
        ticks_labels = [L"\varepsilon_\mathrm{m}", L"\sqrt{\varepsilon}_\mathrm{m}", L"1"]
        cb.set_ticks(ticks)
        cb.set_ticklabels(ticks_labels, fontsize=20)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("After Phase III", fontsize =24)
        fig.savefig("./figures/NormalJacobi_phase3.pdf")
    end
    #Phase 4: Accuracy loss correction
    if offSchur(A) > ε
        normal_jacobi_bunse!(A)
        #println("Done Phase 4 - Accuracy correction")
    end
    if showphase == true
        fig, ax = subplots()
        c = ax.imshow(max.(abs.(A), eps(Float64)), cmap="viridis", aspect="equal", norm=pnorm)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("After Phase IV", fontsize =24)
        fig.savefig("./figures/NormalJacobi_phase4.pdf")
    end
    
    return Tridiagonal(A)
end

normal_skew_jacobi_plots(A::AbstractMatrix{T}) where T = normal_skew_jacobi_plots!(copy(A), true)









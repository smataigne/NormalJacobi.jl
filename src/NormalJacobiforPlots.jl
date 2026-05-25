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
    εₘ = eps(T)                  #Element-wise norm bound
    ε  = εₘ * norm(A) * 10        #Matrix-wise norm bound
    μ  = sqrt(εₘ) * norm(A)                 #Target accuracy for clustering
    ii = zeros(Int64, 2)         #Indices for rows/columns selections
    th = zeros(T, 2, n)
    tv = zeros(T, n, 2)
    iter = 1; itermax = 5 * sqrt(n)
    Ω = zeros(n, n)
    Ω .= (A .- A') / 2
    oldoff = Inf
    offschur = offSchur(Ω)
    #Phase I  Implicit Paardekooper
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
                    ii .= i + 1, j + 1
                    th .= A[ii, :]
                    @. A[i + 1, :] =  c₁ * th[1, :] + s₁ * th[2, :] 
                    @. A[j + 1, :] = -s₁ * th[1, :] + c₁ * th[2, :] 
                    tv .= A[:, ii]
                    @. A[:, i + 1] =  c₁ * tv[:, 1] + s₁ * tv[:, 2]
                    @. A[:, j + 1] = -s₁ * tv[:, 1] + c₁ * tv[:, 2]

                    #Second Similarity transformation "G2' * A * G2"
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
                    ii .= i + 1, j
                    th .= A[ii, :]
                    @. A[i + 1, :] =  c₁ * th[1, :] + s₁ * th[2, :] 
                    @. A[j, :] = -s₁ * th[1, :] + c₁ * th[2, :] 
                    tv .= A[:, ii]
                    @. A[:, i + 1] =  c₁ * tv[:, 1] + s₁ * tv[:, 2]
                    @. A[:, j] = -s₁ * tv[:, 1] + c₁ * tv[:, 2]
                    #Second Similarity transformation "G2' * A * G2"
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
    #Ensures correct signs on Ω
    for i ∈ 1:n-1
        if (A[i + 1, i] - A[i, i + 1]) < 0
            A[:, i + 1] .*= -1
            A[i + 1, :] .*= -1
        end
    end
    #Early stopping
    if offSchur(A) < ε
        return Tridiagonal(A) 
    end
    if showphase == true
        fig, ax = subplots()
        c = ax.imshow(max.(abs.(A), eps(Float64)), cmap="viridis", aspect="equal", norm=pnorm)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("After step I", fontsize =24)
        fig.savefig("./figures/NormalJacobi_phase1.pdf")
    end
    
    adj = build_adjacency(A, μ)
    components = find_connected_components(adj)
    visited = zeros(Bool, length(components))
    println("Connected components: ", components)
    for (i, kk) ∈ enumerate(components)
        if isSSH(A[kk, kk], ε) && !visited[i]
            visited[i] = true
            println("Applying SSH Jacobi on component of size ", length(kk))
            SSHjacobi2!(A, kk)
            #normal_jacobi_bunse2!(A, kk)
            if showphase == true
                fig, ax = subplots()
                c = ax.imshow(max.(abs.(A), eps(Float64)), cmap="viridis", aspect="equal", norm=pnorm)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title("After step II.1", fontsize =24)
                fig.savefig("./figures/NormalJacobi_phase2.pdf")
            end
        end
    end
    for (i, kk) ∈ enumerate(components)
        if issym(A[kk, kk], μ) && !visited[i]
            visited[i] = true
            println("Applying symmetric Jacobi on component of size ", length(kk))
            sym_jacobi2!(A, kk)
            if showphase == true
                fig, ax = subplots()
                c = ax.imshow(max.(abs.(A), eps(Float64)), cmap="viridis", aspect="equal", norm=pnorm)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title("After step II.2", fontsize =24)
                fig.savefig("./figures/NormalJacobi_phase3.pdf")
            end
        end
    end
    for (i, kk) ∈ enumerate(components)
        if !visited[i]
            println("Applying Bunse-Gerstner Jacobi on component of size ", length(kk))
            normal_jacobi_bunse2!(A, kk)
            if showphase == true
                fig, ax = subplots()
                c = ax.imshow(max.(abs.(A), eps(Float64)), cmap="viridis", aspect="equal", norm=pnorm)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title("After step II.3", fontsize =24)
                fig.savefig("./figures/NormalJacobi_phase4.pdf")
            end
        end
    end
    if offSchur(A) > ε
        println("Applying Bunse-Gerstner Jacobi ")
        normal_jacobi_zhou!(A)
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
        ax.set_title("After step III", fontsize =24)
        fig.savefig("./figures/NormalJacobi_phase5.pdf")
    end
    
    return Tridiagonal(A)
end

normal_skew_jacobi_plots(A::AbstractMatrix{T}) where T = normal_skew_jacobi_plots!(copy(A), true)









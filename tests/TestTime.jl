using LinearAlgebra, SkewLinearAlgebra, Plots, LaTeXStrings, BenchmarkTools
include("../src/NormalJacobi.jl")
include("../src/NormalJacobiBunse.jl")
include("../src/NormalJacobiGoldstine.jl")
include("../src/NormalJacobiZhou.jl")
include("../src/Utils.jl")

ns = [10, 32, 100]  # You can add more sizes if needed
K = length(ns)
l1 = zeros(K)
l2 = zeros(K)
l3 = zeros(K)
l4 = zeros(K)
αs = [0.0 0.0; 0.2 0.0; 0.4 0.0; 0.0 0.2; 0.0 0.4; 0.2 0.2]  # You can add more (α, β) pairs if needed
t = size(αs, 1)
for k ∈ 1:t
    α₁ = αs[k, 1]
    α₂ = αs[k, 2]
    for (i, n) ∈ enumerate(ns)
        A = create_matrix(n, α₁, α₂)
        display(norm(A*A'-A'A)/norm(A)^2)
        if k == 1 && i == 1   
            normal_skew_jacobi!(copy(A), false)
            normal_jacobi_bunse!(copy(A))
            normal_jacobi_goldstine!(copy(complex.(A,0)))
            normal_jacobi_zhou!(copy(A))
        end
        l1[i] = @elapsed normal_skew_jacobi!(copy(A), false)
        l2[i] = @elapsed normal_jacobi_bunse!(copy(A))
        l3[i] = @elapsed normal_jacobi_goldstine!(copy(complex.(A,0)))
        l4[i] = @elapsed normal_jacobi_zhou!(copy(A))
        print("ok for n = $n \n")
    end

    P = plot(framestyle=:box, legend=:topleft,font="Computer Modern", titlefont = "Computer Modern",tickfontfamily="Computer Modern",legendfont="Computer Modern", guidefontfamily = "Computer Modern",
    legendfontsize=12,titlefontsize = 16, yguidefontsize=16,xguidefontsize=16, xtickfontsize = 13, ytickfontsize=13, margin = 0.4Plots.cm, minorgrid = false, yscale=:log, xscale=:log)
    plot!(ns, l1, color = :blue, linestyle =:solid, markershape=:circle, label = "Algorithm 1")
    plot!(ns, l2, color = :red, linestyle =:dash, markershape=:diamond, label = "Bunse-Gerstner et al.")
    plot!(ns, l3, color = :green, linestyle =:dot, markershape=:star5, label = "Goldstine et al.")
    plot!(ns, l4, color = :black, linestyle =:dashdot, markershape=:rect, label = "Zhou et al.")
    xlabel!(L"n")   
    ylabel!("Running time [s]")
    title!("Parameters: " * L" $\alpha_1 = $" * string(Int(100α₁))*"%, " * L"$\alpha_2 = $" * string(Int(100α₂)) *"%.")
    savefig(P, "./figures/time_normalskewjacobi_" * string(α₁) * "_" * string(α₂) * ".pdf")
    #display(P)
end

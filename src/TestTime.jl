using LinearAlgebra, SkewLinearAlgebra, Plots, LaTeXStrings, BenchmarkTools
include("NormalJacobi.jl")
include("NormalJacobi2.jl")
include("NormalJacobi4.jl")
include("Utils.jl")

ns = [10, 32, 100, 316]  # You can add more sizes if needed
K = length(ns)
l1 = zeros(K)
l2 = zeros(K)
l3 = zeros(K)
αs = [0.0 0.0; 0.1 0.0; 0.2 0.0; 0.3 0.0; 0.0 0.1; 0.0 0.2; 0.1 0.1; 0.2 0.2]  # You can add more (α, β) pairs if needed
t = size(αs, 1)
for k ∈ 1:t
    α₁ = αs[k, 1]
    α₂ = αs[k, 2]
    for (i, n) ∈ enumerate(ns)
        A = create_matrix(n, α₁, α₂)
        display(norm(A*A'-A'A)/norm(A)^2)
        if k == 1 && i == 1   
            normalskewjacobi!(copy(A), false)
            normaljacobi!(copy(A))
            normaljacobieberlein!(copy(complex.(A,0)))
        end
        l1[i] = @elapsed normalskewjacobi!(copy(A), false)
        l2[i] = @elapsed normaljacobi!(copy(A))
        l3[i] = @elapsed normaljacobieberlein!(copy(complex.(A,0)))
        print("ok for n = $n \n")
    end

    P = plot(framestyle=:box, legend=:topleft,font="Computer Modern", titlefont = "Computer Modern",tickfontfamily="Computer Modern",legendfont="Computer Modern", guidefontfamily = "Computer Modern",
    legendfontsize=12,titlefontsize = 16, yguidefontsize=16,xguidefontsize=16, xtickfontsize = 13, ytickfontsize=13, margin = 0.4Plots.cm, minorgrid = false, yscale=:log, xscale=:log)
    plot!(ns, l1, color = :blue, linestyle =:solid, markershape=:circle, label = L"\texttt{normalskewjacobi}")
    plot!(ns, l2, color = :red, linestyle =:dash, markershape=:diamond, label = L"\texttt{normaljacobi}")
    plot!(ns, l3, color = :green, linestyle =:dot, markershape=:star5, label = L"\texttt{normaljacobieberlein}")
    xlabel!(L"n")   
    ylabel!("Running time [s]")
    title!("Running time comparison for" * L" $\alpha_1 = $" * string(α₁)*", " * L"$\alpha_2 = $" * string(α₂) *".")
    savefig(P, "../figures/time_normalskewjacobi_" * string(α₁) * "_" * string(α₂) * ".pdf")
    display(P)
end

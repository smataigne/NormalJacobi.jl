using LinearAlgebra, SkewLinearAlgebra, Plots, LaTeXStrings, BenchmarkTools
include("../src/NormalJacobi.jl")
include("../src/NormalJacobiBunse.jl")
include("../src/NormalJacobiGoldstine.jl")
include("../src/NormalJacobiZhou.jl")
include("../src/NormalJacobiHe.jl")
include("../src/Utils.jl")
include("./UtilsforTests.jl")


BLAS.set_num_threads(1) # To avoid variability in timings due to multi-threading
ns = [16,32,64,128,256]#, 512]  # You can add more sizes if needed
K = length(ns)
l1 = zeros(K)
l2 = zeros(K)
l3 = zeros(K)
l4 = zeros(K)
l5 = zeros(K)
αs = [0.0 0.0; 0.2 0.2; 0.4 0.4] # You can add more (α, β) pairs if needed
t = size(αs, 1)
N = 10  # Number of trials per case
for k ∈ 1:t
    α₁ = αs[k, 1]
    α₂ = αs[k, 2]
    for (i, n) ∈ enumerate(ns)
        A = create_matrix(n, α₁, α₂)
        if k == 1 && i == 1   
            normal_skew_jacobi!(copy(A), false)
            normal_jacobi_bunse!(copy(A))
            normal_jacobi_goldstine!(copy(complex.(A,0)))
            normal_jacobi_zhou!(copy(A))
            normal_jacobi_he!(copy(A))
        end
        l1[i] = minimum(@elapsed normal_skew_jacobi!(copy(A), false) for _ ∈ 1:N)
        l2[i] = minimum(@elapsed normal_jacobi_bunse!(copy(A)) for _ ∈ 1:N)
        l3[i] = minimum(@elapsed normal_jacobi_goldstine!(copy(complex.(A,0))) for _ ∈ 1:N)
        l4[i] = minimum(@elapsed normal_jacobi_zhou!(copy(A)) for _ ∈ 1:N)
        l5[i] = minimum(@elapsed normal_jacobi_he!(copy(A)) for _ ∈ 1:N)
        print("ok for n = $n \n")
    end
    emin = Int(floor(minimum(log10.(vcat(l1, l2, l3, l4, l5)))))
    emax = Int(ceil(maximum(log10.(vcat(l1, l2, l3, l4, l5)))))
    if iseven(emax-emin) 
        emax += 1
    end
    P = plot(framestyle=:box, legend=:topleft,font="Computer Modern", titlefont = "Computer Modern",tickfontfamily="Computer Modern",legendfont="Computer Modern", guidefontfamily = "Computer Modern",
    legendfontsize=12,titlefontsize = 16, yguidefontsize=16,xguidefontsize=16, xtickfontsize = 13, ytickfontsize=13, margin = 0.4Plots.cm, minorgrid = false, yscale=:log, xscale=:log,  
    xticks = (ns, [latexstring("2^{" * string(i) * "}") for i ∈ 4:9] ), yticks = (10.0 .^ (emin:emax), Base.Iterators.flatten([[latexstring("10^{" * string(i) * "}"), ""]  for i ∈ emin:2:emax])))
    plot!(ns, l1, color = :blue, linestyle =:solid, markershape=:circle, label = "Algorithm 4.4")
    plot!(ns, l2, color = :red, linestyle =:dash, markershape=:diamond, label = "Bunse-Gerstner et al.")
    plot!(ns, l3, color = :green, linestyle =:dot, markershape=:star5, label = "Goldstine et al.")
    plot!(ns, l4, color = :black, linestyle =:dashdot, markershape=:rect, label = "Zhou et al.")
    plot!(ns, l5, color = :orange, linestyle =:dashdotdot, markershape=:utriangle, label = "He et al.")
    plot!(ns[2:end], ((ns[2:end] ./ 32).^3) .* minimum(l1), color = :gray, linestyle =:dash, label = false)
    xlabel!(L"n")   
    ylabel!("Running time [s]")
    ylims!(10.0^(emin-0.1) , 10.0^(emax+0.1))
    title!("Parameters: " * L" $\alpha_1 = $" * string(Int(100α₁))*"%, " * L"$\alpha_2 = $" * string(Int(100α₂)) *"%.")
    savefig(P, "./figures/time_normalskewjacobi_" * string(α₁) * "_" * string(α₂) * ".pdf")
    display(P)

    P = plot(framestyle=:box, legend=:topleft,font="Computer Modern", titlefont = "Computer Modern",tickfontfamily="Computer Modern",legendfont="Computer Modern", guidefontfamily = "Computer Modern",
    legendfontsize=12,titlefontsize = 16, yguidefontsize=16,xguidefontsize=16, xtickfontsize = 13, ytickfontsize=13, margin = 0.4Plots.cm, minorgrid = false, xscale=:log,  
    xticks = (ns, [latexstring("2^{" * string(i) * "}") for i ∈ 4:9] ))
    plot!(ns, l1 ./ l1, color = :blue, linestyle =:solid, markershape=:circle, label = "Algorithm 4.4")
    plot!(ns, l2 ./ l1, color = :red, linestyle =:dash, markershape=:diamond, label = "Bunse-Gerstner et al.")
    plot!(ns, l3 ./ l1, color = :green, linestyle =:dot, markershape=:star5, label = "Goldstine et al.")
    plot!(ns, l4 ./ l1, color = :black, linestyle =:dashdot, markershape=:rect, label = "Zhou et al.")
    plot!(ns, l5 ./ l1, color = :orange, linestyle =:dashdotdot, markershape=:utriangle, label = "He et al.")
    xlabel!(L"n")   
    ylabel!("Relative running time")
    ylims!(0.0 , 1.2*maximum(vcat(l2 ./ l1, l3 ./ l1, l4 ./ l1, l5 ./ l1) ) * 1.05)
    title!("Parameters: " * L" $\alpha_1 = $" * string(Int(100α₁))*"%, " * L"$\alpha_2 = $" * string(Int(100α₂)) *"%.")
    savefig(P, "./figures/time_normalskewjacobi_" * string(α₁) * "_" * string(α₂) * "_relative.pdf")
    display(P)
end
#=
# Final plot for α₁ = 0.2, α₂ = 0.2
α₁ = αs[1, 1]
α₂ = αs[1, 2]
emin = Int(floor(minimum(log10.(vcat(l1, l2, l3, l4, l5)))))
emax = Int(ceil(maximum(log10.(vcat(l1, l2, l3, l4, l5)))))
if iseven(emax-emin) 
    emax += 1
end
P = plot(framestyle=:box, legend=:topleft,font="Computer Modern", titlefont = "Computer Modern",tickfontfamily="Computer Modern",legendfont="Computer Modern", guidefontfamily = "Computer Modern",
legendfontsize=12,titlefontsize = 16, yguidefontsize=16,xguidefontsize=16, xtickfontsize = 13, ytickfontsize=13, margin = 0.4Plots.cm, minorgrid = false, yscale=:log, xscale=:log,  
xticks = (ns, [latexstring("2^{" * string(i) * "}") for i ∈ 4:9] ), yticks = (10.0 .^ (emin:emax), Base.Iterators.flatten([[latexstring("10^{" * string(i) * "}"), ""]  for i ∈ emin:2:emax])))
plot!(ns, l1, color = :blue, linestyle =:solid, markershape=:circle, label = "Algorithm 4.1")
plot!(ns, l2, color = :red, linestyle =:dash, markershape=:diamond, label = "Bunse-Gerstner et al.")
plot!(ns, l3, color = :green, linestyle =:dot, markershape=:star5, label = "Goldstine et al.")
plot!(ns, l4, color = :black, linestyle =:dashdot, markershape=:rect, label = "Zhou et al.")
plot!(ns, l5, color = :orange, linestyle =:dashdotdot, markershape=:utriangle, label = "He et al.")
plot!(ns[2:end], ((ns[2:end] ./ 32).^3) .* minimum(l1), color = :gray, linestyle =:dash, label = false)
xlabel!(L"n")   
ylabel!("Running time [s]")
ylims!(10.0^(emin-0.1) , 10.0^(emax+0.1))
title!("Parameters: " * L" $\alpha_1 = $" * string(Int(100α₁))*"%, " * L"$\alpha_2 = $" * string(Int(100α₂)) *"%.")
savefig(P, "./figures/time_normalskewjacobi_" * string(α₁) * "_" * string(α₂) * ".pdf")
display(P)

P = plot(framestyle=:box, legend=:topleft,font="Computer Modern", titlefont = "Computer Modern",tickfontfamily="Computer Modern",legendfont="Computer Modern", guidefontfamily = "Computer Modern",
legendfontsize=12,titlefontsize = 16, yguidefontsize=16,xguidefontsize=16, xtickfontsize = 13, ytickfontsize=13, margin = 0.4Plots.cm, minorgrid = false, xscale=:log,  
xticks = (ns, [latexstring("2^{" * string(i) * "}") for i ∈ 4:9] ))
plot!(ns, l1 ./ l1, color = :blue, linestyle =:solid, markershape=:circle, label = "Algorithm 4.1")
plot!(ns, l2 ./ l1, color = :red, linestyle =:dash, markershape=:diamond, label = "Bunse-Gerstner et al.")
plot!(ns, l3 ./ l1, color = :green, linestyle =:dot, markershape=:star5, label = "Goldstine et al.")
plot!(ns, l4 ./ l1, color = :black, linestyle =:dashdot, markershape=:rect, label = "Zhou et al.")
plot!(ns, l5 ./ l1, color = :orange, linestyle =:dashdotdot, markershape=:utriangle, label = "He et al.")
xlabel!(L"n")   
ylabel!("Relative running time")
ylims!(0.0 , 1.5*maximum(vcat(l2 ./ l1, l3 ./ l1, l4 ./ l1, l5 ./ l1) ) * 1.05)
title!("Parameters: " * L" $\alpha_1 = $" * string(Int(100α₁))*"%, " * L"$\alpha_2 = $" * string(Int(100α₂)) *"%.")
savefig(P, "./figures/time_normalskewjacobi_" * string(α₁) * "_" * string(α₂) * "_relative.pdf")
display(P)
=#



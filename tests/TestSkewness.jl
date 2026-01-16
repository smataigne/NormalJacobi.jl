using LinearAlgebra, BenchmarkTools, Plots, LaTeXStrings
include("../src/Utils.jl")
include("../src/NormalJacobi.jl")
include("../src/NormalJacobiZhou.jl")
include("../src/NormalJacobiBunse.jl")
include("../src/NormalJacobiGoldstine.jl")
include("../src/NormalJacobiHe.jl")


n = 100
A = Matrix(qr(randn(n, n)).Q)
εₘ = eps(Float64)
H = (A + A') / 2
Ω = (A - A') / 2
Ω .*= norm(H) / norm(Ω)

η = 0.31622776601683794  # approx 10^-0.5
K = ceil(Int, log(1e-6) / log(η))
exponents = 0:K
L = length(exponents)
l1 = zeros(L)
l2 = zeros(L)
l3 = zeros(L)
l4 = zeros(L)
l5 = zeros(L)

a1 = zeros(L)
a2 = zeros(L)
a3 = zeros(L)
a4 = zeros(L)
a5 = zeros(L)

for (i, k) ∈ enumerate(exponents)
    γ = η^k
    A = (1 - γ) * H + γ * Ω

    B = copy(A)
    normal_skew_jacobi!(B, false)
    a1[i] = offSchur(B) / norm(A)
    B = copy(A)
    normal_jacobi_bunse!(B)
    a2[i] = offSchur(B) / norm(A)
    B = complex.(copy(A), 0)
    normal_jacobi_goldstine!(B)
    a3[i] = offdiag(B) / norm(A)
    B = copy(A)
    normal_jacobi_zhou!(B)
    a4[i] = offSchur(B) / norm(A)
    B = complex.(copy(A), 0)
    normal_jacobi_he!(B)
    a5[i] = offdiag(B) / norm(A)

    T = 10
    
    l1[i] = minimum(@elapsed normal_skew_jacobi!(copy(A), false) for _ in 1:T) 
    l2[i] = minimum(@elapsed normal_jacobi_bunse!(copy(A)) for _ in 1:T) 
    l3[i] = minimum(@elapsed normal_jacobi_goldstine!(copy(complex.(A,0))) for _ in 1:T) 
    l4[i] = minimum(@elapsed normal_jacobi_zhou!(copy(A)) for _ in 1:T) 
    l5[i] = minimum(@elapsed normal_jacobi_he!(copy(complex.(A,0))) for _ in 1:T)
    
    print("ok for k = $k \n")
    print(norm((A-A')/2)/norm(A), "\n")
end

γ = η .^ exponents
τ = sqrt.(γ .^2 ./ (γ .^2 .+ (1 .- γ) .^2))

ms = 3
P = plot(framestyle=:box, legend=:topright,font="Computer Modern", titlefont = "Computer Modern",tickfontfamily="Computer Modern",legendfont="Computer Modern", guidefontfamily = "Computer Modern",
legendfontsize=12,titlefontsize = 16, yguidefontsize=16,xguidefontsize=16, xtickfontsize = 13, ytickfontsize=13, margin = 0.4Plots.cm, minorgrid = false, xscale=:log
)
plot!(τ, l1, color = :blue, linestyle =:solid, markershape=:circle, markersize = ms, label = "Algorithm 5.3")
plot!(τ, l2, color = :red, linestyle =:dash, markershape=:diamond, markersize = ms, label = "Bunse-Gerstner et al.")
plot!(τ, l3, color = :green, linestyle =:dot, markershape=:star5,  markersize = ms, label = "Goldstine et al.")
plot!(τ, l4, color = :black, linestyle =:dashdot, markershape=:rect, markersize = ms, label = "Zhou et al.")
plot!(τ, l5, color = :orange, linestyle =:dashdotdot, markershape=:utriangle, markersize = ms, label = "He et al.")
xlabel!(L"\Vert \mathrm{skew}(A)\  \Vert_\mathrm{F} \ /\ \Vert A\ \Vert_\mathrm{F}")   
ylabel!("Running time [s]")
ylims!(-0.05 , 2.05)
savefig(P, "./figures/test_skewness.pdf")
display(P)
P2 = plot(framestyle=:box, legend=:topleft,font="Computer Modern", titlefont = "Computer Modern",tickfontfamily="Computer Modern",legendfont="Computer Modern", guidefontfamily = "Computer Modern",
legendfontsize=12,titlefontsize = 16, yguidefontsize=16,xguidefontsize=16, xtickfontsize = 13, ytickfontsize=13, margin = 0.4Plots.cm, minorgrid = false, xscale=:log, yscale=:log, yticks = ([εₘ, 10εₘ, 100εₘ, 1000εₘ], [L"\varepsilon_\mathrm{m}", L"10\varepsilon_\mathrm{m}", L"100\varepsilon_\mathrm{m}", L"1000\varepsilon_\mathrm{m}"])
)
plot!(τ, max.(a1, εₘ), color = :blue, linestyle =:solid, markershape=:circle, markersize = ms, label = "Algorithm 5.3")
plot!(τ, max.(a2, εₘ) , color = :red, linestyle =:dash, markershape=:diamond, markersize = ms, label = "Bunse-Gerstner et al.")
plot!(τ, max.(a3, εₘ), color = :green, linestyle =:dot, markershape=:star5,  markersize = ms, label = "Goldstine et al.")
plot!(τ, max.(a4, εₘ), color = :black, linestyle =:dashdot, markershape=:rect, markersize = ms, label = "Zhou et al.")
plot!(τ, max.(a5, εₘ), color = :orange, linestyle =:dashdotdot, markershape=:utriangle, markersize = ms, label = "He et al.")
xlabel!(L"\Vert \mathrm{skew}(A)\   \Vert_\mathrm{F} \ /\ \Vert A\ \Vert_\mathrm{F}")   
ylabel!(L"\mathrm{offschur}(A\ )\ /\ \Vert A\ \Vert_\mathrm{F}")
ylims!(0.8εₘ , 1300εₘ)
savefig(P2, "./figures/test_skewness_offschur.pdf")
display(P2)


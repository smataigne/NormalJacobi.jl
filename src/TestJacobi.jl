using LinearAlgebra, BenchmarkTools, Plots, LaTeXStrings
include("Jacobi.jl")
BLAS.set_num_threads(1)
#=
ns = [10, 32, 100, 316, 1000]
N = length(ns)
t1 = zeros(N)
t2 = zeros(N)
for (i,n) ∈ enumerate(ns)
    A = randn(n, n)
    B = Matrix(qr(A).Q)
    A = (B + B') + 10 * (B - B')
    #@time NormalJacobi!(copy(A))
    #@time normaljacobi2!(copy(A))
    t1[i] = @belapsed normaljacobi($A)
    t2[i] = @belapsed normalskewjacobi($A)
    print("ok\n")
end
=#
P = plot(framestyle=:box, legend=:topleft,font="Computer Modern", tickfontfamily="Computer Modern",legendfont="Computer Modern", guidefontfamily = "Computer Modern",
legendfontsize=10,yguidefontsize=17,xguidefontsize=17, xtickfontsize = 13, ytickfontsize=13,margin = 0.3Plots.cm, minorgrid = false, yscale=:log, xscale=:log)
plot!(ns, t1, color = :blue, linestyle =:solid, markershape=:circle, label = L"\texttt{normaljacobi}")
plot!(ns, t2, color = :red, linestyle =:dash, markershape=:diamond, label = L"\texttt{normalskewjacobi}")
xlabel!(L"n")
ylabel!("Running time [s]")
savefig(P, "time_normalskewjacobi.pdf")
display(P)
P = plot(framestyle=:box, legend=:topright,font="Computer Modern", tickfontfamily="Computer Modern",legendfont="Computer Modern", guidefontfamily = "Computer Modern",
legendfontsize=10,yguidefontsize=17,xguidefontsize=17, xtickfontsize = 13, ytickfontsize=13,margin = 0.3Plots.cm, minorgrid = false, xscale=:log)
plot!(ns, t1 ./ t2, color = :blue, linestyle =:solid, markershape=:circle, label = L"\texttt{normaljacobi}")
plot!(ns, t2 ./ t2, color = :red, linestyle =:dash, markershape=:diamond, label = L"\texttt{normalskewjacobi}")
xlabel!(L"n")
ylabel!("Running time ratio")
savefig(P, "relative_time_normalskewjacobi.pdf")
display(P)

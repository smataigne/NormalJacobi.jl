using LinearAlgebra, SkewLinearAlgebra, LaTeXStrings, BenchmarkTools, PyPlot, PyCall

PyPlot.matplotlib.rc("text", usetex=true)
PyPlot.matplotlib.rc("font", family="serif", serif=["Computer Modern"])
PyPlot.matplotlib.rc("mathtext", fontset="cm")

A = 10 .^(- 16* rand(20,20))

norm = PyCall.pyimport("matplotlib.colors").LogNorm(
    vmin = eps(Float64),
    vmax = 1
)
fig, ax = subplots()
c = ax.imshow(max.(abs.(A), eps(Float64)), cmap="viridis", aspect="equal", norm=norm)

ticks = [eps(Float64), sqrt(eps(Float64)), 1.0]
ticks_labels = [L"\varepsilon_\mathrm{m}", L"\sqrt{\varepsilon}_\mathrm{m}", L"1"]

ax.set_xticks([])
ax.set_yticks([])
ax.set_title("After Phase III", fontsize=20)
display(fig)
fig.savefig("matrix_plot.pdf")
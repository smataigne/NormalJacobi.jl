using LinearAlgebra, Plots, LaTeXStrings

m = 10
U1 = Matrix(qr(randn(m, m)).Q)
U2 = Matrix(qr(randn(m, m)).Q)
V1 = Matrix(qr(randn(m, m)).Q)
V2 = Matrix(qr(randn(m, m)).Q)
Θ = Diagonal(π .* rand(m))
C1 = cos(Θ)
S1 = sin(Θ)

A = U1 * C1 * V1' + U2 * C1 * V2' 
B = U2 * S1 * V1' + U1 * S1 * V2'
M = [A; B]


O = complex.(A, B)
svdO = svd(O)
P = svdO.U * svdO.Vt
R = [real(P); imag(P)]
display(tr(M'R))

function f(U::AbstractMatrix, V::AbstractMatrix, C::AbstractMatrix, S::AbstractMatrix) 
    global U1, U2, V1, V2, C1, S1
    M1 = U * C * V'
    M2 = U * S * V'
    M3 = U1 * C1 * V1'+ U2 * C1 * V2'
    M4 = U2 * S1 * V1'+ U1 * S1 * V2'
    return tr(M3'M1 + M4'M2)
end

function find_nearest_ortho_symplectic_old()
    global U1, U2, V1, V2, C1, S1
    iter = 1
    itermax = 1000
    ε = 1e-6
    fs = zeros(itermax)
    M1 = U1 * C1 * V1' + U2 * C1 * V2'
    M2 = U2 * S1 * V1' + U1 * S1 * V2'
    C = copy(C1)
    S = copy(S1)

    #Initial Guess
    SVD_U = svd(U1 + U2)
    U = SVD_U.U * SVD_U.V'
    SVD_V = svd(V1 + V2)
    V = SVD_V.U * SVD_V.V'

    fs[1] = f(U, V, C, S)
    while (iter == 1 || abs(fs[iter] - fs[iter - 1]) > ε) && iter < itermax
        #Optimize U
        SVD_U = svd(C * V' * M1' + S * V' * M2')
        U = SVD_U.V * SVD_U.U'
        #Optimize V
        SVD_V = svd(M1' * U * C + M2' * U * S)
        V = SVD_V.U * SVD_V.V'
        iter += 1
        fs[iter] = f(U, V, C, S)
    end
    return U, V, fs[1:iter]
end

function find_nearest_ortho_symplectic()
    global U1, U2, V1, V2, C1, S1
    iter = 1
    itermax = 1000
    ε = 1e-6
    fs = zeros(itermax)
    M1 = U1 * C1 * V1' + U2 * C1 * V2'
    M2 = U2 * S1 * V1' + U1 * S1 * V2'
    C = copy(C1)
    S = copy(S1)

    #Initial Guess
    SVD_U = svd(U1 + U2)
    U = SVD_U.U * SVD_U.V'
    SVD_V = svd(V1 + V2)
    V = SVD_V.U * SVD_V.V'

    fs[1] = f(U, V, C, S)
    while (iter == 1 || abs(fs[iter] - fs[iter - 1]) > ε) && iter < itermax
        #Optimize U
        SVD_U = svd(C * V' * M1' + S * V' * M2')
        U = SVD_U.V * SVD_U.U'
        #Optimize V
        SVD_V = svd(M1' * U * C + M2' * U * S)
        V = SVD_V.U * SVD_V.V'
        iter += 1
        fs[iter] = f(U, V, C, S)
        #Optimize C and S
        A = V' * M1' * U
        B = V' * M2' * U
        for i ∈ 1:m
            α = A[i, i]
            β = B[i, i] 
            if β > 0 
                C[i, i] = sign(α / β) * abs(α) / hypot(α, β) #sqrt((α^2) / (α^2 + β^2))
                S[i, i] = abs(β) / hypot(α, β)  #sqrt((β^2) / (α^2 + β^2))
            else
                C[i, i] = sign(α)
                S[i, i] = 0.0 
            end
        end
    end
    return U, V, C, S, fs[1:iter]
end

U, V, C, S, fs = find_nearest_ortho_symplectic()
U_opt2, V_opt2, fs2 = find_nearest_ortho_symplectic_old()
println("Done")
display(fs[end])
P = plot(framestyle=:box, legend=:bottomright,font="Computer Modern", titlefont = "Computer Modern",tickfontfamily="Computer Modern",legendfont="Computer Modern", guidefontfamily = "Computer Modern",
legendfontsize=12,titlefontsize = 16, yguidefontsize=16,xguidefontsize=16, xtickfontsize = 13, ytickfontsize=13, margin = 0.4Plots.cm, minorgrid = false
)
plot!(1:length(fs), fs, marker=:circle, linestyle=:solid, label = L"C,S" * " optimized", xlabel="Iteration", ylabel="Objective function", title= "Convergence of Nearest Ortho-Symplectic")
plot!(1:length(fs2), fs2, marker=:diamond, linestyle=:dash, label = L"C,S" *  " fixed")
savefig(P, "./figures/convergence_nearest_ortho_symplectic.pdf")
display(P)

R = [U * C * V' -U * S * V'; U * S * V' U * C * V']
display(tr(M'R[:, 1:m]))
display(norm(R'R - I(2m)))



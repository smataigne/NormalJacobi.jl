using LinearAlgebra, Plots

p = 40
A = randn(p, p)
B = randn(p, p)
A = (A - A')/2

Θ = randn(2p, 2p)
Θ = (Θ - Θ')/2
Q = exp(Θ)
if det(Q) < 0
    Q[:, 1] .= -Q[:, 1]
end
K = 50
l1 = zeros(K)
Γ = zeros(p, p)
for k ∈ 1:K
    Q .= [Q[:, 1:p] Q[:, p+1:2p] * exp(Γ)]
    M = log(Q)
    M = (M - M')/2
    Γ .= -M[p+1:2p, p+1:2p]
    l1[k] = norm(M[p+1:2p, p+1:2p])
end
Θ = randn(2p, 2p)
Θ = (Θ - Θ')/2
Q = exp(Θ)
if det(Q) < 0
    Q[:, 1] .= -Q[:, 1]
end
l2 = zeros(K)
Γ .= zeros(p, p)
for k ∈ 1:K
    Q .= [Q[:, 1:p] Q[:, p+1:2p] * exp(Γ)]
    Q .= [Q[1:p, :]; exp(Γ) * Q[p+1:2p, :]]
    M = log(Q)
    M = (M - M')/2
    Γ .= -M[p+1:2p, p+1:2p]/2
    l2[k] = norm(M[p+1:2p, p+1:2p])
end
P = plot(yscale=:log)
plot!(1:K, l1)
plot!(1:K, l2)
display(P)
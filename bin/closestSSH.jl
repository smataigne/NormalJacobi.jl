using LinearAlgebra

m = 4
A = randn(m, m)
B = randn(m, m)
C = randn(m, m)
D = randn(m, m)

M = [A B; C D]

sym(M::AbstractMatrix) = (M + M') / 2
skew(M::AbstractMatrix) = (M - M') / 2

P = [sym((A + D)/2)  skew((B - C)/2);
     skew((C - B)/2)  sym((A + D)/2)]
R = M - P
for k ∈   1:10
    W = randn(m, m)
    X = randn(m, m)
    W = W + W'
    X = X - X'
    S = [W -X; X W]
    display(tr(R'S))
end

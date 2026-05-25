using LinearAlgebra

n = 4
A = Matrix(qr(randn(2n, 2n)).Q)
display(det(A))
if det(A) < 0
    A[:, 1] = -A[:, 1]
end
#SVD = svd(A - A')
#S = SVD.U * SVD.V'
S = A * sqrt(- A' * A')
V = randn(2n, n)
V2 = Matrix(qr((S + 1im * I(2n)) * V).Q)
Q2 = sqrt(2) * [real.(V2) imag.(V2)]
display(round.(Q2' * A * Q2; digits=3))
B= Q2' * A * Q2
U = B[1:n,1:n] + 1im * B[1:n, n .+ (1:n)]
display(round.(U'U; digits=3))
t = randn()
c = cos(t)
s = sin(t)
G = [c -s; s c]
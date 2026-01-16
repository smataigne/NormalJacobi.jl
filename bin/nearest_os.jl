using   LinearAlgebra


m = 3

A = randn(2m, 2m)
A ./= norm(A) * sqrt(2m)

J = [zeros(m, m)  I(m);
     -I(m)      zeros(m, m)]
A1 = (A  + J * A * J') / 2
A2 = (A  - J * A * J') / 2

B = complex.(A1[1:m, 1:m], A1[m+1:2m, 1:m])
svdB = svd(B)
P = svdB.U * svdB.Vt
R = [real(P) -imag(P); imag(P) real(P)]

display(norm(A1 - R)^2 + norm(A2)^2)
display(norm(A - R)^2)

display(2 * norm(B - P)^2 + norm(A2)^2)
display(norm(A2)^2 + 0.25 * (norm(A'A - I) + norm(A'J*A - J))^2)

display(2 * norm(B'B- I)^2)
display(norm(A1'A1 - I)^2)
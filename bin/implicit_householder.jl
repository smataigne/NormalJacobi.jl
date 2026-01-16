using LinearAlgebra


@views function householder(u)
    v = similar(u, size(u))
    α = sign(u[1])
    v .= u
    v[1] = u[1] - α * norm(u)
    return v ./ norm(v)
end

@views function implicit_householder!(A::AbstractMatrix)
    n = size(A, 1)
    for i ∈ 1:n-2
        v = householder(0.5 *(A[i+1:end, i] - reshape(A[i,i+1:end], (n-i,1))))
        A[i+1:end, :] = (I - 2*v*v') * A[i+1:end, :]
        A[:, i+1:end] = A[:, i+1:end]  * (I - 2*v*v')'
    end 
    return A
end

n = 16
A = Matrix(qr(randn(n,n)).Q)
if det(A) < 0
    A[:,1] .= -A[:,1]
end
A = implicit_householder!(A)
kk = [1:2:n; 2:2:n]
display(A[kk, kk])
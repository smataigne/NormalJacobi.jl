using LinearAlgebra

"""
    create_matrix(n::Integer, α₁::Number, α₂::Number) -> Matrix{Float64}
Create an n x n real matrix with specified proportions of real and complex eigenvalues.
The parameters α₁ and α₂ determine the fractions of real and repeated complex eigenvalues, respectively
"""
function create_matrix(n::Integer, α₁::Number, α₂::Number)
    Q = Matrix(qr(randn(n, n)).Q)
    d = zeros(n)
    sd = zeros(n - 1) 
    a₁ = floor(Int, α₁ * n)
    if isodd(a₁)  && iseven(n)  # Ensure a₁ is even if n is even
        a₁ -= 1
    end
    a₂ = floor(Int, α₂ * n)
    if isodd(a₂)  
        a₂ -= 1
    end
    s = randn(a₂ ÷ 2)
    d[1:a₁] = randn(a₁)
    d[(a₁+1):(a₁+a₂)] = invpermute!([s; s], [1:2:a₂;2:2:a₂])
    sd[(a₁+1):2:(a₁+a₂ - 1)] .= randn()
    p = n - a₁ - a₂
    s = randn(p ÷ 2)
    d[(a₁ + a₂ + 1):end] = invpermute!([s; s], [1:2:p;2:2:p])
    sd[(a₁ + a₂ + 1):2:end] .= randn(p÷2)#1 .+  0.01*√(eps(Float64))  * randn(p÷2)
    return Matrix(Q * Tridiagonal(sd, d, -sd) * Q')
end
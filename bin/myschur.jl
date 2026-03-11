using LinearAlgebra

@views function myschur!(A::AbstractMatrix{T}, Z::AbstractMatrix{T}) where T
    H = hessenberg!(A)
    lmul!(H.Q, Z)
    return LAPACK.hseqr!('S', 'V', 1, size(A, 1), A, Z)[3]
end

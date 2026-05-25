using LinearAlgebra, SkewLinearAlgebra, LaTeXStrings
include("Utils.jl")
include("NormalJacobiZhou.jl")
include("NormalJacobiBunse.jl")
include("UtilsJacobi.jl")

"""
    normal_jacobi_skew!(A::AbstractMatrix{T}) where T

Performs a Jacobi-like algorithm for normal matrices in-place on `A`. 
The input matrix `A` is modified to be tridiagonal with 2x2 or 1x1 blocks on the diagonal. 
The function returns the tridiagonal form of `A`.
"""
@views function normal_jacobi_skew!(A::AbstractMatrix{T}) where T
    n = size(A, 1)
    εₘ = eps(T)                  #Element-wise norm bound
    ε  = εₘ * norm(A) * 10        #Matrix-wise norm bound
    μ  = sqrt(εₘ * norm(A))       #Target accuracy for clustering
    #Step I.1 Implicit Paardekooper
    implicit_paardekooper!(A)
    #Early stopping
    if offschur(A) < ε
        return Tridiagonal(A) 
    end
    #Step I.2 Build the adjacency matrix for clustering.
    adj = build_adjacency(A, μ)
    #Step I.3 Clustering
    components = find_connected_components(adj)
    for kk ∈ components
        if length(kk) > 2
            if is_ssh(A[kk, kk], 10ε)
                #Step II.1
                ssh_jacobi2!(A, kk)
            elseif is_sym(A[kk, kk], μ)
                #Step II.2
                symmetric_jacobi2!(A, kk)
            else
                #Step II.3
                normal_jacobi_bunse2!(A, kk)
            end
        end
    end
    if offschur(A) > ε
        #Step III 
        normal_jacobi_zhou!(A)
    end
    return Tridiagonal(A)
end

normal_jacobi_skew(A::AbstractMatrix) = normal_jacobi_skew!(copy(A))









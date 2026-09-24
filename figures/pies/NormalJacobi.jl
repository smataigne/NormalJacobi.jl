using LinearAlgebra, SkewLinearAlgebra, LaTeXStrings
include("Utils.jl")
include("NormalJacobiZhou.jl")
include("NormalJacobiBunse.jl")
include("UtilsJacobi.jl")

"""
    time_normal_jacobi_skew!(A::AbstractMatrix{T}) where T

Performs a Jacobi-like algorithm for normal matrices in-place on `A`. 
The input matrix `A` is modified to be tridiagonal with 2x2 or 1x1 blocks on the diagonal. 
The function returns the tridiagonal form of `A`.
"""
@views function time_normal_jacobi_skew!(A::AbstractMatrix{T}) where T
    n = size(A, 1)
    εₘ = eps(T)                  #Element-wise norm bound
    ε  = εₘ * norm(A) * 10        #Matrix-wise norm bound
    μ  = sqrt(εₘ * norm(A))       #Target accuracy for clustering
    #Step I.1 Implicit Paardekooper
    tI1 = @elapsed begin
        implicit_paardekooper!(A)
        #Early stopping
        if offschur(A) < ε
            return Tridiagonal(A) 
        end
    end
    #Step I.2 Build the adjacency matrix for clustering.
    tI2 = @elapsed begin
        adj = build_adjacency(A, μ)
    end
    #Step I.3 Clustering
    tI3 = @elapsed begin
        components = find_connected_components(adj)
    end
    tII = @elapsed begin
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
    end
    tIII = @elapsed begin
        if offschur(A) > ε
            #Step III 
            normal_jacobi_zhou!(A)
        end
    end
    return Tridiagonal(A), [tI1+tI2+tI3, tII, tIII]
end

time_normal_jacobi_skew(A::AbstractMatrix) = time_normal_jacobi_skew!(copy(A))









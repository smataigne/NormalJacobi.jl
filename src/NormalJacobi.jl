using LinearAlgebra, SkewLinearAlgebra, LaTeXStrings
include("Utils.jl")
include("NormalJacobiZhou.jl")
include("NormalJacobiBunse.jl")
include("UtilsJacobi.jl")

@views function normal_jacobi_skew!(A::AbstractMatrix{T}) where T
    n = size(A, 1)
    εₘ = eps(T)                  #Element-wise norm bound
    ε  = εₘ * norm(A) * 10        #Matrix-wise norm bound
    μ  = sqrt(εₘ * norm(A))       #Target accuracy for clustering
    #Phase I  Implicit Paardekooper
    implicit_paardekooper!(A)
    #Early stopping
    if offschur(A) < ε
        return Tridiagonal(A) 
    end
    adj = build_adjacency(A, μ)
    components = find_connected_components(adj)
    #println("Connected components: ", components)
    for kk ∈ components
        if length(kk) > 2
            if is_ssh(A[kk, kk], μ)
                #println("Applying SSH Jacobi on component of size ", length(kk))
                ssh_jacobi2!(A, kk)
            elseif is_sym(A[kk, kk], μ)
                #println("Applying symmetric Jacobi on component of size ", length(kk))
                symmetric_jacobi2!(A, kk)
            else
                #println("Applying Bunse-Gerstner Jacobi on component of size ", length(kk))
                normal_jacobi_bunse2!(A, kk)
            end
        end
    end
    if offschur(A) > ε
        #println("Applying Zhou Jacobi ")
        normal_jacobi_zhou!(A)
    end
    return Tridiagonal(A)
end

normal_jacobi_skew(A::AbstractMatrix) = normal_jacobi_skew!(copy(A))









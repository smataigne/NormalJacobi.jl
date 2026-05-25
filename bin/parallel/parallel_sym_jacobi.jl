using LinearAlgebra, BenchmarkTools, OhMyThreads
include("../src/Utils.jl")
include("../src/UtilsJacobi.jl")
include("../src/ParallelUtilsJacobi.jl")
using Base.Threads

@views function parallel_symmetric_jacobi3!(A::AbstractMatrix{T}, kk::AbstractVector{Int}, temp::AbstractMatrix) where T
    K = length(kk)
    n = size(A, 1)
    iter = 1
    itermax = min(5 * sqrt(K), 20)
    old_offdiag = Inf
    new_offdiag = sym_offdiag(A[kk, kk])
    ε = eps(T) * 10 * norm(A)
    εₘ = eps(T) * 10
    steps = parallel_cyclic_order(K)
    do_it = zeros(Bool, n)
    cs = zeros(n)
    ss = zeros(n)
    nthreads = Threads.nthreads()
    while new_offdiag > ε && iter < itermax && new_offdiag < old_offdiag
        for k1 ∈ 1:K-1
            Threads.@threads for tt ∈ 1:nthreads
                kmin = (tt - 1) * (K ÷ 2) ÷ nthreads + 1
                kmax = tt * (K ÷ 2) ÷ nthreads
                for k2 ∈ kmin:kmax
                    i = min(steps[k1, k2, 1], steps[k1, k2, 2])
                    j = max(steps[k1, k2, 1], steps[k1, k2, 2])
                    ii = kk[i]
                    jj = kk[j]
                    r = (A[jj, ii] + A[ii, jj]) / 2
                    if abs(r) > εₘ
                        do_it[ii] = true
                        c, s = jacobi_rotation(A[ii, ii], r, A[jj, jj])
                        cs[ii] = c
                        ss[ii] = s
                        temp[ii, :] .= A[ii, :]
                        temp[jj, :] .= A[jj, :]
                        @. A[ii, :] =  cs[ii] * temp[ii, :] + ss[ii] * temp[jj, :]
                        @. A[jj, :] = -ss[ii] * temp[ii, :] + cs[ii] * temp[jj, :]
                    else
                        do_it[ii] = false
                    end
                end
            end
            Threads.@threads for tt ∈ 1:nthreads
                kmin = (tt - 1) * (K ÷ 2) ÷ nthreads + 1
                kmax = tt * (K ÷ 2) ÷ nthreads
                for k2 ∈ kmin:kmax
                    i = min(steps[k1, k2, 1], steps[k1, k2, 2])
                    j = max(steps[k1, k2, 1], steps[k1, k2, 2])
                    ii = kk[i]
                    jj = kk[j]
                    if do_it[ii]
                        temp[:, ii] .= A[:, ii]
                        temp[:, jj] .= A[:, jj]
                        @. A[:, ii] =  cs[ii] * temp[:, ii] + ss[ii] * temp[:, jj]
                        @. A[:, jj] = -ss[ii] * temp[:, ii] + cs[ii] * temp[:, jj]
                        do_it[ii] = false
                    end
                end
            end
        end
        iter += 1
        old_offdiag = new_offdiag
        new_offdiag = sym_offdiag(A[kk, kk])
        #(new_offdiag)
    end
    return A
end
BLAS.set_num_threads(1)
#=
n = 16
A = Matrix(qr(randn(n, n)).Q)
temp = zeros(n, n)
parallel_implicit_paardekooper!(copy(A), copy(temp))
implicit_paardekooper!(copy(A))
n = 600
A = Matrix(qr(randn(n, n)).Q)
temp = zeros(n, n)
@time implicit_paardekooper!(copy(A))
@time parallel_implicit_paardekooper!(copy(A), copy(temp))
=#

n = 32
A = randn(n, n)
A = A + A'
temp = zeros(n, n)
symmetric_jacobi2!(copy(A), Array(1:n))
parallel_symmetric_jacobi3!(copy(A), Array(1:n), temp)
n = 1024
A = randn(n, n)
A = A + A'
temp = zeros(n, n)
#@time symmetric_jacobi2!(copy(A), Array(1:n))
@time parallel_symmetric_jacobi3!(copy(A), Array(1:n), temp)
println("Done")
#=
@btime parallel_symmetric_jacobi2!(copy(A), Array(1:n))
@btime parallel_symmetric_jacobi2!(copy(A), Array(1:n))
=#
@views function parallel_normal_jacobi_skew!(A::AbstractMatrix{T}) where T
    n = size(A, 1)
    εₘ = eps(T)                  #Element-wise norm bound
    ε  = εₘ * norm(A) * 10        #Matrix-wise norm bound
    μ  = sqrt(εₘ * norm(A))       #Target accuracy for clustering
    #Phase I  Implicit Paardekooper
    parallel_implicit_paardekooper!(A)
    #Early stopping
    if offschur(A) < ε
        return Tridiagonal(A) 
    end
    adj = build_adjacency(A, μ)
    components = find_connected_components(adj)
    println("Connected components: ", components)
    for kk ∈ components
        if length(kk) > 2
            if is_ssh(A[kk, kk], 10ε)
                println("Applying SSH Jacobi on component of size ", length(kk))
                parallel_ssh_jacobi2!(A, kk)
            elseif is_sym(A[kk, kk], μ)
                println("Applying symmetric Jacobi on component of size ", length(kk))
                parallel_symmetric_jacobi2!(A, kk)
            else
                println("Applying Bunse-Gerstner Jacobi on component of size ", length(kk))
                parallel_normal_jacobi_bunse2!(A, kk)
            end
        end
    end
    if offschur(A) > ε
        println("Applying Zhou Jacobi")
        parallel_normal_jacobi_zhou!(A)
    end
    return Tridiagonal(A)
end

parallel_normal_jacobi_skew(A::AbstractMatrix) = parallel_normal_jacobi_skew!(copy(A))


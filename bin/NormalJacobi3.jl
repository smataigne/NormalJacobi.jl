using LinearAlgebra

"""
    offdiag(A::AbstractMatrix)

Compute the off-diagonal Frobenius norm of a symmetric matrix A.
"""
function offdiag(A::AbstractMatrix{<:Complex})
    Σ = 0
    n = size(A, 1)
    for i ∈ 1:n
        for j ∈ (i+1):n
            Σ += abs(A[i, j])^2 + abs(A[j, i])^2
        end
    end 
    return Σ
end

function findmax(A::AbstractMatrix{<:Complex})
    n = size(A, 1)
    maxval = 0.0
    maxi, maxj = 1, 1
    for i ∈ 1:n
        for j ∈ (i+1):n
            if i != j && abs(A[i, j]) > maxval
                maxval = abs(A[i, j])
                maxi, maxj = i, j
            end
        end
    end
    return maxi, maxj
end
"""
    optimaljacobi(A::AbstractMatrix{<:Complex})
Perform one iteration of the optimal Jacobi method for normal matrices.
Input:  - a normal matrix `A`.\\
"""
@views function optimaljacobi!(A::AbstractMatrix{<:Complex})
    n = size(A, 1)
    T = eltype(A)
    ε = 100 * eps(real(T)) * norm(A)
    δ = offdiag(A)
    iter = 0
    itermax = 1
    ii = zeros(Int, 2)
    temp1 = zeros(T, 2, n)
    temp2 = zeros(T, n, 2)
    while δ > ε && iter < itermax
        #for i ∈ 1:n-1
            #for j ∈ i+1:n
                i, j = findmax(A)
                if i == j
                    continue
                else
                    
                    b = (A[i, j] + A[j, i]') / 2
                    ω = (A[i, j] - A[j, i]') / 2 
                    r = abs(b)
                    β = angle(b)#atan(imag(b), real(b))#
                    s = abs(ω)
                    γ = angle(ω)#atan(imag(ω), real(ω))#angle(ω)
                    ν = (iszero(r * s) ? π/2 : β - γ)
                    u = abs(real(A[j, j])) - abs(real(A[i, i]))
                    v = abs(imag(A[j, j])) - abs(imag(A[i, i]))
                    t = complex(u, v)
                    #print("t :", t ,"\n")
                    #OK 2*(r^2+s^2) = abs(A[i,j])^2 + abs(A[j,i])^2)
                    if iszero(t)
                        #Determine α
                        d = sqrt((r^2 + s^2)^2 * sin(ν)^2 + (r^2 - s^2)^2 * cos(ν)^2)
                        sμ = - (r^2 + s^2) * sin(ν) / d
                        cμ = (r^2 - s^2) * cos(ν) / d
                        μ = atan(sμ, cμ)
                        α = 0.5 * (β + γ - μ)
                        #Determine ϕ
                        ϕ = 0.25 * acos(sign(κ))     
                    else
                        L = u * v - 4 * r * s * sin(ν)
                        M = u^2 - v^2 + 4 * (r^2 - s^2)
                        #display((L, M))
                        Ar = L * (r^2 - s^2) * sin(ν) + M * r * s 
                        B  = L * (r^2 + s^2) * cos(ν)
                        C  = L * (r^2 - s^2) + M * r * s * sin(ν)
                        #display(Ar^2 + B^2 - C^2)
                        #display(r^2*s^2*(M^2 + 2*L^2)*cos(ν)^2)
                        #display((Ar, B, C))
                        if !iszero(Ar) && !iszero(B)
                        #display((Ar,B, C))
                            cμ = (B * C + sign(cos(ν)) * Ar * sqrt(Ar^2 + B^2 - C^2)) / (Ar^2 + B^2)
                            sμ = (- Ar* C + sign(cos(ν)) * B * sqrt(Ar^2 + B^2 - C^2)) / (Ar^2 + B^2)
                            μ = atan(sμ , cμ)
                            α = (β + γ - μ) / 2
                            display((β/π, γ/π, μ/π, α/π))
                            κ = u^2 + v^2 - 4 * (r^2 * cos(β - α)^2 + s^2 * sin(γ - α)^2)
                            λ = 4 * (r * u * cos(β - α) + s * v * sin(γ - α))
        
                            tϕ = - λ / κ
                            cϕ = κ / sqrt(κ^2 + λ^2)
                            sϕ = tϕ * cϕ
                            ϕ = 0.25 * atan(sϕ, cϕ)
                            #display((α, ϕ))
                        
                        end
                        #x = 2 / t * complex(r * cos(β - α), s * sin(γ - α))
                        #y = 2 / t * complex(s * cos(γ - α), r * sin(β - α))
                        #OK (κ, λ) = ((abs(t)^2 * (1-abs(x)^2), abs(t)^2 *(x + x'))
                        #k = 2 * r *s *(sin(μ) + sin(ν))/(L + 2 *r *s * (sin(μ) + sin(ν)))
                        #print("k :", k,"\n")
                        #print("L : ", L,"\n")
                        #print(sin(μ) + sin(ν), "\n")
                        #########
                        α = 0.1
                        ϕ = 0.5
                        κ = u^2 + v^2 - 4 * (r^2 * cos(β - α)^2 + s^2 * sin(γ - α)^2)
                        λ = 4 * (r * u * cos(β - α) + s * v * sin(γ - α))
                        #######
                        q = complex(cos(α), sin(α))
                        U = [cos(ϕ) (sin(ϕ) * q); (-q' * sin(ϕ)) cos(ϕ)]
                        ii .= i, j
                        #display(U)
                        oldδ = offdiag(A)
                        temp1 .= A[ii, :]
                        mul!(A[ii, :], U, temp1, 1, 0)
                        temp2 .= A[:, ii]
                        mul!(A[:, ii], temp2, U', 1, 0)
                        δ = offdiag(A)
                        #display((κ, λ))
                        #OK, =0, display(κ * cos(4ϕ) - λ * sin(4ϕ) - sqrt(κ^2 + λ^2))
                        #print("(ϕ, α)=", (ϕ, α), "\n")
                        #print(Ar*sin(μ),"  ",B*cos(μ) - C , "\n")
                        display(δ - oldδ)
                        display(0.5 * κ * sin(2ϕ)^2 + 0.25 * λ * sin(4ϕ))
                        b = (A[i, j] + A[j, i]') / 2
                        ω = (A[i, j] - A[j, i]') / 2 
                        nr = abs(b)
                        ns = abs(ω)
                        #display(2 * (nr^2 + ns^2)- 2 * (r^2 + s^2))

                        #display(0.25 * (κ - sqrt(κ^2 + λ^2)))
                        #display(0.25 * (u^2 + v^2 - 4 * (r^2 + s^2) - sqrt(M^2 + 4 * L^2)))

                        #display(A[ii, ii])
                    end
                end
            #end
        #end
        iter += 1
        δ = offdiag(A)
        display(δ)
    end
end

n = 6
A = Matrix(qr(randn(ComplexF64, n , n)).Q)
#A = 0.5 * (A + A') + 2 * (A - A')
#A = complex.(A, 0*A)
#display(A)
optimaljacobi!(A)















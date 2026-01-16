#=
    i = 1
    while i < n
        if abs(Σ[i]) > ε && !solved[i]
            l[1] = i  
            l[2] = i + 1
            nk = 2  
            solved[i:i+1] .= true
            for j ∈ i+1:n-1
                if abs(Σ[j] - Σ[i]) < μ && !solved[j]
                    #push!(kk, j, j + 1)
                    nk += 2
                    l[nk - 1] = j  
                    l[nk] = j + 1
                    solved[j:j+1] .= true
                end
            end
            #nk = length(kk)
            kk = l[1:nk]
            if nk > 2
                normal_jacobi_bunse!(A[kk, kk])
            else
                solved[i:i+1] .= false
            end
        end
        i += 1
    end
    =#
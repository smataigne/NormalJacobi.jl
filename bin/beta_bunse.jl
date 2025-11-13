using LinearAlgebra

n = 4
ii = zeros(Integer, 2)
tv = zeros(Float64, n, 2)
th = zeros(Float64, 2, n)
Q = Matrix(qr(randn(Float64, n , n)).Q)
num = Q[2,2] * Q[3, 1] - Q[2,1] * Q[3,2]
den = Q[1,2] * Q[2, 1] - Q[1,1] * Q[2,2] 
θ₁ = atan(num , den)
c₁, s₁ = cos(θ₁), sin(θ₁)
display( - c₁ * num + den * s₁)
ii .= 1, 3
th .= Q[ii, :]
@. Q[1, :] =  c₁ * th[1, :] + -s₁ * th[2, :] 
@. Q[3, :] =  s₁ * th[1, :] + c₁ * th[2, :]
display(Q[2,1] * Q[3,2] - Q[2,2] * Q[3, 1])
θ₂ = atan( - Q[3, 2], Q[2, 2])
c₂, s₂ = cos(θ₂), sin(θ₂)
ii .= 2, 3
th .= Q[ii, :]
@. Q[2, :] =  c₂ * th[1, :] + -s₂ * th[2, :] 
@. Q[3, :] =  s₂ * th[1, :] + c₂ * th[2, :]
display(Q[3,1])
display(Q[3, 2])
num = Q[2,2] * Q[4, 1] - Q[2,1] * Q[4,2]
den = Q[1,2] * Q[2, 1] - Q[1,1] * Q[2,2] 
θ₃ = atan(num , den)
c₃, s₃ = cos(θ₃), sin(θ₃)
ii .= 1, 4
th .= Q[ii, :]
@. Q[1, :] =  c₃ * th[1, :] + -s₃ * th[2, :]
@. Q[4, :] =  s₃ * th[1, :] + c₃ * th[2, :]
display( - c₃ * num + den * s₃)
display(Q[2, 1] * Q[4,2] - Q[2,2] * Q[4, 1])
θ₄ = atan( - Q[4, 2], Q[2, 2])
c₄, s₄ = cos(θ₄), sin(θ₄)
ii .= 2, 4  
th .= Q[ii, :]
@. Q[2, :] =  c₄ * th[1, :] + -s₄ * th[2, :]
@. Q[4, :] =  s₄ * th[1, :] + c₄ * th[2, :]
display(Q)


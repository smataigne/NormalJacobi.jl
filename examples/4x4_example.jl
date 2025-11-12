using LinearAlgebra

roundmatrix(A::AbstractMatrix) = round.(A; digits=3)
#Initialization of a 4x4 example matrix
A = [1  1  1 -1; 
     1  1 -1  1; 
     1 -1 -1  -1; 
     1 -1  1  1]

Ω = (A - A') / 2

####Phase 1: Applying Paardekooper's method to A
#Step 1 of Paardekooper's method
G₁ = [-√2/2  0 √2/2  0;
       0     0   0   1;
      -√2/2  0 -√2/2 0;
       0    -1   0   0]

A₁ = G₁' * A * G₁
Ω₁ = (A₁ - A₁') / 2
print("After first step of Paardekooper's method:\n")
print("A₁ = \n")
display(roundmatrix(A₁))
print("Ω₁ = \n")
display(roundmatrix(Ω₁))

#Step 2 of Paardekooper's method
G₂ = [√6/3  0    0   -√3/3;
       0    1    0    0;
       0    0    1    0;
      √3/3 0   0   √6/3]
A₂ = G₂' * A₁ * G₂
Ω₂ = (A₂ - A₂') / 2
print("After second step of Paardekooper's method:\n")
print("A₂ = \n")
display(roundmatrix(A₂))
print("Ω₂ = \n")
display(roundmatrix(Ω₂))

#End of Paardekooper's method
#Phase 2: Applying the classical Jacobi method to A₂
G₃= [1 0 0 0;
     0 1 0 0;
     0 0 1/2 -√3/2;
     0 0 √3/2 1/2]
A₃ = G₃' * A₂ * G₃
Ω₃ = (A₃ - A₃') / 2
print("After applying one step of the classical Jacobi method:\n")
print("A₃ = \n")
display(roundmatrix(A₃))
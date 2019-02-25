# a!/b!
factorial_ratio(a::I, b::I) where {I<:Integer} = gamma(a+1)/gamma(b+1) #binomial(a,b)*factorial(a-b)

function coulomb_analytical(k::I, γ::R, ℓ::I, r̃::R) where {I<:Integer, R<:Real}
    e⁻ᵞʳ = exp(-γ*r̃)
    
    # S = -gamma(k+ℓ+1)/r̃^(ℓ+1)*(e⁻ᵞʳ-1)/γ^(k+ℓ+1)
    S = 2gamma(k+ℓ+1)*exp(-γ*r̃/2 -(ℓ+1)*log(r̃))*sinh(γ*r̃/2)/γ^(k+ℓ+1)
    
    for i ∈ 0:k+ℓ-1
        S -= factorial_ratio(k+ℓ,k+ℓ-i)*r̃^(k-i-1)*e⁻ᵞʳ/γ^(i+1)
    end
    
    for i ∈ 0:k-ℓ-1
        S += factorial_ratio(k-ℓ-1,k-ℓ-i-1)*r̃^(k-i-1)*e⁻ᵞʳ/γ^(i+1)
    end
    
    S
end

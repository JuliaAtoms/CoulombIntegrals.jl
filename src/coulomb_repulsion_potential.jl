struct CoulombRepulsionPotential{PP<:AbstractPoissonProblem,
                                 Potential, Diag, Vec}
    poisson::PP
    V̂::Potential
    r⁻¹::Diag
    tmp::Vec
end

function CoulombRepulsionPotential(R::CompactBases.BasisOrRestricted,
                                   poisson::AbstractPoissonProblem)
    V̂ = LinearOperator(DiagonalOperator(applied(*, R, poisson.Y)), R)
    r = axes(R,1)
    r⁻¹ = R'*QuasiDiagonal(1 ./ r)*R
    tmp = zeros(eltype(poisson), size(R,2))
    CoulombRepulsionPotential(poisson, V̂, LinearOperator(r⁻¹, R), tmp)
end

function CoulombRepulsionPotential(R::CompactBases.BasisOrRestricted,
                                   k::Number, ::Type{C}=eltype(R)) where C
    poisson = PoissonProblem(R, k, C)
    CoulombRepulsionPotential(R, poisson)
end

function Base.copyto!(potential::CoulombRepulsionPotential, ρ::Density)
    solve!(potential.poisson, ρ)
    copyto!(potential.V̂.A, potential.poisson.Y)
    potential
end

function LinearAlgebra.mul!(y, potential::CoulombRepulsionPotential, x,
                            α::Number=true, β::Number=false)
    mul!(y, potential.r⁻¹,
         mul!(potential.tmp, potential.V̂, x),
         α, β)
end

export CoulombRepulsionPotential

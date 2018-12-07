using FEDVRQuasi
using FiniteDifferencesQuasi
using LinearAlgebra
using IterativeSolvers
import ContinuumArrays: axes
import ContinuumArrays.QuasiArrays: AbstractQuasiMatrix, MulQuasiArray

function poisson!(w::MulQuasiArray{T,N,<:Mul{<:Tuple,<:Tuple{<:B,<:A}}},
                  ρ::MulQuasiArray{T,N,<:Mul{<:Tuple,<:Tuple{<:B,<:A}}},
                  ℓ::I; kwargs...) where {T,N,B<:AbstractQuasiMatrix{T},A<:AbstractArray,I<:Integer}
    axes(w) == axes(ρ) || throw(DimensionMismatch("Incompatible axes"))
    R,ρc = ρ.mul.factors
    Rw,wc = w.mul.factors

    R == Rw || throw(DimensionMismatch("Incompatible bases"))

    D = Derivative(axes(R,1))
    Tm = R'D'D*R
    Tm *= -1
    r = locs(R)
    Tm += Diagonal(ℓ*(ℓ+1)./r.^2)

    v = ρc./r

    cg!(wc, Tm, v; kwargs...)
    wc .*= (2ℓ+1)

    # Homogeneous contribution
    rℓ = R*(r.^ℓ)
    s = (ρ'rℓ)[1]/(r[end]^(2ℓ+1))
    wc .+= s*r.^(ℓ+1)

    wc ./= r

    w
end

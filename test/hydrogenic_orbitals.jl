using LinearAlgebra
using ArnoldiMethod
import ContinuumArrays.QuasiArrays: AbstractQuasiMatrix

using Polynomials
using OrthoPoly

function orb_label((n,ℓ))
    ℓs = "spdfg"[ℓ+1]
    "$n$(ℓs)"
end

"""
    hydredwfn(n,ℓ[,Z=1])

Returns a function of `r` (in atomic units, i.e. Bohr radii) that
evaluates the _reduced_ hydrogenic radial orbital corresponding to the
quantum numbers `n` and `ℓ`, and for a nuclear charge of `Z`.
"""
function hydredwfn(::Type{Tuple}, n,ℓ,Z=1)
    n > 0 || throw(ArgumentError("Invalid n = $n"))
    0 ≤ ℓ && ℓ<n || throw(ArgumentError("Invalid ℓ = $ℓ"))

    N = √((2Z/n)^3*factorial(n-ℓ-1)/(2n*factorial(n+ℓ)))

    ρ = Polynomials.Poly([0,2Z/n],:r)
    R = ρ^ℓ*laguerre_assoc(n-ℓ-1,2ℓ+1,ρ)
    P = Polynomials.Poly([0,1],:r)*R

    N,ρ,P
end

"""
    hydredwfn(n,ℓ[,Z=1])

Returns a function of `r` (in atomic units, i.e. Bohr radii) that
evaluates the _reduced_ hydrogenic radial orbital corresponding to the
quantum numbers `n` and `ℓ`, and for a nuclear charge of `Z`.
"""
function hydredwfn(n,ℓ,Z=1)
    N,ρ,P = hydredwfn(Tuple,n,ℓ,Z)

    r -> N*polyval(P,r)*exp(-polyval(ρ,r)/2)
end

function get_orbitals(R::B, ℓ::Int, Z, nev::Int,
                      mode::Symbol) where {B<:AbstractQuasiMatrix}
    if mode == :arnoldi
        println("Finding eigenstates for Z = $Z, ℓ = $ℓ")
        Tℓ = CoulombIntegrals.get_double_laplacian(R,ℓ)
        Tℓ ./= 2
        V = Matrix(r -> -Z/r, R)
        H = Tℓ + V

        schur,history = partialschur(H, nev=nev, tol=sqrt(eps()), which=SR())
        println(history)
        println(diag(schur.R))

        [(n+ℓ,ℓ) => normalize!(R*(schur.Q[:,n]*sign(schur.Q[1,n]))) for n = 1:nev]
    elseif mode == :symbolic
        r = CoulombIntegrals.locs(R)
        [(n,ℓ) => R*hydredwfn(n,ℓ,Z).(r)
         for n ∈ ℓ .+ (1:nev)]
    else
        throw(ArgumentError("Unknown mode $(mode)"))
    end
end

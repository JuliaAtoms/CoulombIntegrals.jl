using ArnoldiMethod
import ContinuumArrays.QuasiArrays: AbstractQuasiMatrix
using LazyArrays

using ClassicalOrthogonalPolynomials

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
function hydredwfn(n,ℓ,Z=1)
    n > 0 || throw(ArgumentError("Invalid n = $n"))
    0 ≤ ℓ && ℓ<n || throw(ArgumentError("Invalid ℓ = $ℓ"))

    q = 2Z/n
    N = √(q^3*factorial(n-ℓ-1)/(2n*factorial(n+ℓ)))

    r -> N*r*(q*r)^ℓ*laguerrel(n-ℓ-1,2ℓ+1,q*r)*exp(-q*r/2)
end

function get_orbitals(R::B, ℓ::Int, Z, nev::Int,
                      mode::Symbol) where {T,B<:AbstractQuasiMatrix{T}}
    if mode == :arnoldi
        println("Finding eigenstates for Z = $Z, ℓ = $ℓ")
        Tℓ = CoulombIntegrals.get_double_laplacian(R,ℓ,T)
        Tℓ ./= -2
        r = axes(R,1)
        H = Tℓ + R' * QuasiDiagonal(-Z ./ r) * R

        schur,history = partialschur(H, nev=nev, tol=sqrt(eps()), which=SR())
        println(history)
        println(diag(schur.R))

        [(n+ℓ,ℓ) => normalize!(applied(*, R, (schur.Q[:,n]*sign(schur.Q[1,n])))) for n = 1:nev]
    elseif mode == :symbolic
        r = axes(R,1)
        [(n,ℓ) => applied(*, R, R \ hydredwfn(n,ℓ,Z).(r))
         for n ∈ ℓ .+ (1:nev)]
    else
        throw(ArgumentError("Unknown mode $(mode)"))
    end
end

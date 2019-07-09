abstract type AbstractPoissonProblem end

mutable struct PoissonProblem{T,U,B<:AbstractQuasiMatrix,
                              V<:AbstractVector{U},
                              M<:AbstractMatrix, LD,
                              RV<:RadialOrbital{U,B},
                              RO₁<:RadialOrbital{T,B},
                              RHS<:AbstractVector,
                              RO₂<:RadialOrbital{T,B},
                              RO₃<:RadialOrbital{T,B},
                              Factorization} <: AbstractPoissonProblem
    k::Int
    r⁻¹::V
    rᵏ::RV
    rᵏ⁺¹::V
    rₘₐₓ⁻²ᵏ⁺¹::U
    Tᵏ::M # Laplacian
    uv::LD # Lazy mutual density
    ρ::RO₁ # Mutual density
    ∫ρ::T # Integrated mutual density
    rhs::RHS # Right-hand side of Poisson problem
    y::RO₂ # Intermediate solution
    w′::RO₃ # Solution
    Tᵏ⁻¹::Factorization # Factorized Laplacian
end

"""
    get_double_laplacian(R, k)

Return Laplacian (multiplied by `2`) for partial wave `k` of the
basis `R`, `-∂ᵣ² + k(k+1)/r²`.
"""
function get_double_laplacian(R::B,k::I,::Type{T}) where {B<:AbstractQuasiMatrix,I<:Integer,T}
    D = Derivative(axes(R,1))
    Tᵏ = R' * D' * D * R
    r = locs(R)
    Tᵏ *= -1
    V = Matrix(r -> k*(k+1)/r^2, R) # Is this correct for any basis? E.g. banded in B-splines?
    Tᵏ += V
    isreal(T) ? Tᵏ : complex(Tᵏ)
end

# LinearAlgebra.isposdef(T::Union{Tridiagonal,SymTridiagonal}) = isposdef(Matrix(T))

"""
    PoissonProblem(k, u, v[; w′=similar(u), Tᵏ=get_double_laplacian(R,k)])

Create the Poisson problem of order `k` for the mutual density `u†(r)
.* v(r)`. `w′` is a `MulQuasiVector` of the same kind as `u` and `v`,
and may optionally be provided as a pre-allocated vector, in case it
is e.g. the diagonal of a potential matrix. The same way, the
Laplacian may be reused from other Poisson problems of the same order,
but with different orbitals `u` and/or `v`.
"""
function PoissonProblem(k::Int, u::RO₁, v::RO₂;
                        w′::RO₃=similar(u),
                        Tᵏ::M = get_double_laplacian(u.args[1],k,T),
                        kwargs...) where {T,B<:AbstractQuasiMatrix,
                                          RO₁<:RadialOrbital{T,B},
                                          RO₂<:RadialOrbital{T,B},
                                          RO₃<:RadialOrbital{T,B},
                                          M<:AbstractMatrix}
    axes(u) == axes(v) || throw(DimensionMismatch("Incompatible axes"))
    Ru,cu = u.args
    Rv,cv = v.args
    Ru == Rv || throw(DimensionMismatch("Incompatible bases"))

    ρ = similar(u)
    rhs = similar(u.args[2])

    y=similar(w′)
    # Strong zero to get rid of random NaNs
    y.args[2] .= false

    Tᵏ⁻¹ = factorization(Tᵏ; kwargs...)

    R = w′.args[1]
    r = locs(R)

    rₘₐₓ = rightendpoint(axes(R,1).domain)

    r = locs(R)
    # Vandermonde matrix for interpolating functions
    RV = R[r,:]
    r⁻¹ = inv.(r)
    rᵏ = RV \ r.^k
    rᵏ⁺¹ = RV \ r.^(k+1)

    PoissonProblem(k, r⁻¹,
                   R ⋆ rᵏ, rᵏ⁺¹, inv(rₘₐₓ^(2k+1)),
                   Tᵏ, u .⋆ v, ρ, zero(T), rhs, y, w′, Tᵏ⁻¹)
end

#=
This method computes the integral \(Y^k(n\ell,n'\ell;r)/r\) where
\[\begin{aligned}
Y^k(n\ell,n'\ell;r)&\equiv
r\int_0^\infty
U^k(r,s)
P^*(n\ell;s)
P(n'\ell';s)
\mathrm{d}s\\
&=
\int_0^r
\left(\frac{s}{r}\right)^k
P^*(n\ell;s)
P(n'\ell';s)
\mathrm{d}s+
\int_r^\infty
\left(\frac{r}{s}\right)^{k+1}
P^*(n\ell;s)
P(n'\ell';s)
\mathrm{d}s
\end{aligned}\]
through the solution of Poisson's problem
\[\left[\frac{\mathrm{d}^2}{\mathrm{d}r^2} -
 \frac{k(k+1)}{r^2}\right]
Y^k(n\ell,n'\ell;r)
=-\frac{2k+1}{r}
P^*(n\ell;r)P(n'\ell';r)\]
subject to the boundary conditions
\[\begin{cases}
Y^k(n\ell,n'\ell';0) &= 0,\\
\frac{\mathrm{d}}{\mathrm{d}r}
Y^k(n\ell,n'\ell';r) &=
-\frac{k}{r}
Y^k(n\ell,n'\ell';r),\quad
r \to \infty.
\end{cases}\]

References:

- Fischer, C. F., & Guo, W. (1990). Spline Algorithms for the
  Hartree-Fock Equation for the Helium Ground State. Journal of
  Computational Physics, 90(2),
  486–496. http://dx.doi.org/10.1016/0021-9991(90)90176-2

- McCurdy, C. W., Baertschy, M., & Rescigno, T. N. (2004). Solving the
  Three-Body Coulomb Breakup Problem Using Exterior Complex
  Scaling. Journal of Physics B: Atomic, Molecular and Optical
  Physics, 37(17),
  137–187. http://dx.doi.org/10.1088/0953-4075/37/17/r01

=#

# Ugly work-around until Poisson problem is properly expressed in a
# basis-agnostic way.
function weightit!(w::RadialOrbital{T,B}) where {T,B<:FEDVRQuasi.BasisOrRestricted{<:FEDVR}}
    R,wc = w.args
    R′ = FEDVRQuasi.unrestricted_basis(R)
    a,b = FEDVRQuasi.restriction_extents(R)
    wc .*= R′.n[1+a:end-b]
    w
end
weightit!(w::RadialOrbital) = w

function (pp::PoissonProblem)(lazy_density=pp.uv; verbosity=0, io::IO=stdout, kwargs...)
    k,ρ,r⁻¹ = pp.k,pp.ρ,pp.r⁻¹
    R,ρc = ρ.args
    wc = pp.w′.args[2]
    yc = pp.y.args[2]

    copyto!(ρ, lazy_density) # Form density

    pp.rhs .= (2k+1) * ρc .* r⁻¹
    ldiv!(yc, pp.Tᵏ⁻¹, pp.rhs)

    copyto!(wc, yc)

    # Add in homogeneous contribution
    pp.∫ρ = materialize(applied(*, ρ', pp.rᵏ))
    s = pp.∫ρ*pp.rₘₐₓ⁻²ᵏ⁺¹
    wc .+= s*pp.rᵏ⁺¹

    wc .*= r⁻¹

    weightit!(pp.w′)

    pp.w′
end

# For exchange potentials, where the density is formed in part from
# the orbital which is "acted upon".
function (pp::PoissonProblem)(v::RO; kwargs...) where {RO<:RadialOrbital}
    R,vc = v.args
    pp.uv.R == R ||
        throw(DimensionMismatch("Cannot form mutual density from different bases"))
    pp(applied(*, R, pp.uv.u) .⋆ v; kwargs...)
end

mutable struct AsymptoticPoissonProblem{T,U,B₁,B₂,
                                        PP<:PoissonProblem{T,U,B₁},
                                        I<:AbstractRange,
                                        RO₁<:RadialOrbital{T,B₂},
                                        VO₂, VO₃} <: AbstractPoissonProblem
    pp::PP # Poisson problem of the inner region
    R̃::B₂ # Basis of the inner region
    inner::I # Range of inner region
    w′::RO₁ # Solution
    w′tail::VO₂ # View of the asymptotic part of w′
    w̃::VO₃ # Asymptotic part of the solution
end

"""
    AsymptoticPoissonProblem(k, u, v, R̃[; w′=similar(U)])

Create the Poisson problem of order `k` for the mutual density `u†(r)
.* v(r)`; the Poisson problem is solved numerically within the domain
of `R̃` and an asymptotic solution is used outside. For this to be
valid, `u` or `v` have to vanish before the end of `R̃`. `w′` is a
`MulQuasiVector` of the same kind as `u` and `v`, and may optionally
be provided as a pre-allocated vector, in case it is e.g. the diagonal
of a potential matrix.
"""
function AsymptoticPoissonProblem(k::Int, u::RO₁, v::RO₂,
                                  R̃::AbstractQuasiMatrix;
                                  w′::RO₃=similar(u),
                                  kwargs...) where {T,B<:AbstractQuasiMatrix,
                                                    RO₁<:RadialOrbital{T,B},
                                                    RO₂<:RadialOrbital{T,B},
                                                    RO₃<:RadialOrbital{T,B}}
    uc = u.args[2]
    vc = u.args[2]
    R,w′c = w′.args
    # It is assumed that grid spacings &c agree.
    axes(R̃,1).domain ⊆ axes(R,1).domain &&
        axes(R̃,2) ⊆ axes(R,2) ||
        throw(ArgumentError("$(R̃) not a subset of $(R)"))
    inner = 1:size(R̃,2)
    tail = inner[end]+1:size(R,2)
    r = locs(R)[tail]

    pp = PoissonProblem(k,
                        applied(*, R̃, view(uc, inner)),
                        applied(*, R̃, view(vc, inner));
                        w′ = applied(*, R̃, view(w′c, inner)),
                        kwargs...)

    w̃ = inv.(r.^(k+1))
    AsymptoticPoissonProblem(pp, R̃, inner, w′, view(w′c, tail), w̃)
end

function (app::AsymptoticPoissonProblem)(lazy_density; kwargs...)
    u = applied(*, app.R̃, view(lazy_density.u, app.inner))
    v = applied(*, app.R̃, view(lazy_density.v, app.inner))
    app.pp(u .⋆ v)

    # This is useful for the case we the asymptotic tail is not
    # actually needed, e.g. when applied to an orbital of compact
    # support.
    if !isempty(app.w̃)
        # Copy over asymptotic solution and weight it by the charge
        # density within the inner region.
        app.w′tail .= app.pp.∫ρ .* app.w̃
    end

    app.w′
end

export AbstractPoissonProblem, PoissonProblem, AsymptoticPoissonProblem

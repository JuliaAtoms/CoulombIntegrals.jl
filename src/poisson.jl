abstract type AbstractPoissonProblem end

mutable struct PoissonProblem{C,InvLaplacian,Diag,Vec,Yt,Metric} <: AbstractPoissonProblem
    k::C
    T⁻¹::InvLaplacian # Factorized Laplacian
    r⁻¹::Diag
    rhs::Vec # Right-hand side of Poisson problem
    Y::Yt
    rᵏ::Vec
    rᵏ⁺¹::Vec
    rₘₐₓ⁻²ᵏ⁺¹::C
    S::Metric
    ∫ρ::C # Integrated mutual density
end

Base.eltype(::PoissonProblem{C}) where C = C

isrealtype(::Type{T}) where {T<:Real} = true
isrealtype(::Type{T}) where {T<:Complex} = false

"""
    get_double_laplacian(R, k)

Return Laplacian (multiplied by `2`) for partial wave `k` of the
basis `R`, `-∂ᵣ² + k(k+1)/r²`.
"""
function get_double_laplacian(R,k,::Type{T}) where T
    D = Derivative(axes(R,1))
    Tᵏ = apply(*, R', D', D, R)
    r = axes(R,1)
    if Tᵏ isa BlockSkylineMatrix
        Tᵏ = Matrix(Tᵏ)
    end
    V = apply(*, R', QuasiDiagonal(k*(k+1)./r.^2), R)
    Tᵏ -= V
    isrealtype(T) ? Tᵏ : complex(Tᵏ)
end

struct PoissonCache{M,F,Diag,V₁,V₂}
    T::M
    T⁻¹::F
    r⁻¹::Diag
    rᵏ::V₁
    rᵏ⁺¹::V₂
end

function get_or_create_poisson_cache!(poisson_cache, k, R, ::Type{C}) where C
    k in keys(poisson_cache) && return poisson_cache[k]
    T = get_double_laplacian(R,k,C)
    T⁻¹ = factorize(T isa BlockSkylineMatrix ? Matrix(T) : T)
    r = axes(R,1)

    f = isrealtype(C) ? identity : complex
    r⁻¹ = f(R'*QuasiDiagonal(1 ./ r)*R)
    rᵏ   = f(R \ r.^k)
    rᵏ⁺¹ = f(R \ r.^(k+1))

    pc = PoissonCache(T, T⁻¹, r⁻¹, rᵏ, rᵏ⁺¹)
    poisson_cache[k] = pc
    pc
end

"""
    PoissonProblem(R, k[, ::Type{C}=eltype(R); poisson_cache])

Create the Poisson problem of order `k` for densities expanded over
`R`.
"""
function PoissonProblem(R, k, ::Type{C}=eltype(R);
                        Y=zeros(C, size(R,2)),
                        poisson_cache::Dict{Int,<:PoissonCache} = Dict{Int,PoissonCache}()) where C
    pc = get_or_create_poisson_cache!(poisson_cache, k, R, C)

    r = axes(R,1)
    rₘₐₓ⁻²ᵏ⁺¹ = inv(rightendpoint(r.domain)^(2k+1))

    PoissonProblem(C(k), pc.T⁻¹, pc.r⁻¹, similar(Y), Y,
                   pc.rᵏ, pc.rᵏ⁺¹, C(rₘₐₓ⁻²ᵏ⁺¹), R'R, zero(C))
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

function solve!(poisson::PoissonProblem, ρ::AbstractVector)
    mul!(poisson.rhs, poisson.r⁻¹, ρ, -(2poisson.k+1), false)
    ldiv!(poisson.Y, poisson.T⁻¹, poisson.rhs)

    # Add in homogeneous contribution
    poisson.∫ρ = dot(poisson.rᵏ, poisson.S, ρ)
    s = poisson.∫ρ*poisson.rₘₐₓ⁻²ᵏ⁺¹
    poisson.Y .+= s .* poisson.rᵏ⁺¹
end

solve!(poisson::PoissonProblem, ρ::Density) =
    solve!(poisson, ρ.ρ)

mutable struct AsymptoticPoissonProblem{PP<:PoissonProblem, B,
                                        I<:AbstractRange,
                                        VO₁, VO₂, VO₃} <: AbstractPoissonProblem
    pp::PP # Poisson problem of the inner region
    R̃::B # Basis of the inner region
    inner::I # Range of inner region
    Y::VO₁ # Solution
    Ytail::VO₂ # View of the asymptotic part of Y′
    Ỹ::VO₃ # Asymptotic part of the solution
end

"""
    AsymptoticPoissonProblem(R, k, u, v, R̃[; w′=similar(U)])

Create the Poisson problem of order `k` for the mutual density `u†(r)
.* v(r)`; the Poisson problem is solved numerically within the domain
of `R̃` and an asymptotic solution is used outside. For this to be
valid, `u` or `v` have to vanish before the end of `R̃`. `w′` is a
`MulQuasiVector` of the same kind as `u` and `v`, and may optionally
be provided as a pre-allocated vector, in case it is e.g. the diagonal
of a potential matrix.
"""
function AsymptoticPoissonProblem(R, k::Int,
                                  R̃::AbstractQuasiMatrix,
                                  ::Type{C}=eltype(R);
                                  kwargs...) where C
    r = axes(R, 1)
    # It is assumed that grid spacings &c agree.
    axes(R̃,1).domain ⊆ r.domain &&
        axes(R̃,2) ⊆ axes(R,2) ||
        throw(ArgumentError("$(R̃) not a subset of $(R)"))
    inner = 1:size(R̃,2)
    tail = inner[end]+1:size(R,2)

    Y = zeros(C, size(R,2))
    pp = PoissonProblem(R̃, k, C; Y=view(Y, inner), kwargs...)

    Ỹ = (R \ r .^ (-k))[tail]

    AsymptoticPoissonProblem(pp, R̃, inner, Y, view(Y, tail), Ỹ)
end

function solve!(poisson::AsymptoticPoissonProblem, ρ::Density)
    solve!(poisson.pp, view(ρ.ρ, poisson.inner))

    # This is useful for the case we the asymptotic tail is not
    # actually needed, e.g. when applied to an orbital of compact
    # support.
    if !isempty(poisson.Ỹ)
        # Copy over asymptotic solution and weight it by the charge
        # density within the inner region.
        poisson.Ytail .= poisson.pp.∫ρ .* poisson.Ỹ
    end

    poisson.Y
end

export AbstractPoissonProblem, PoissonProblem, solve!, AsymptoticPoissonProblem

struct PoissonProblem{T,U,B<:AbstractQuasiMatrix,
                      V<:AbstractVector{U},
                      M<:AbstractMatrix, LD,
                      RV<:RadialOrbital{U,B},
                      RO₁<:RadialOrbital{T,B},RO₂<:RadialOrbital{T,B},
                      Factorization<:LaplacianFactorization}
    k::Int
    r⁻¹::V
    rᵏ::RV
    rᵏ⁺¹::V
    rₘₐₓ⁻²ᵏ⁺¹::U
    Tᵏ::M # Laplacian
    uv::LD # Lazy mutual density
    ρ::RO₁ # Mutual density
    w′::RO₂ # Solution
    Tᵏ⁻¹::Factorization # Factorized Laplacian
end

"""
    get_double_laplacian(R, k)

Return Laplacian (multiplied by `2`) for partial wave `k` of the
basis `R`, `-∂ᵣ² + k(k+1)/r²`.
"""
function get_double_laplacian(R::B,k::I) where {B<:AbstractQuasiMatrix,I<:Integer}
    D = Derivative(axes(R,1))
    Tᵏ = R'D'D*R
    r = locs(R)
    Tᵏ *= -1
    V = Matrix(r -> k*(k+1)/r^2, R) # Is this correct for any basis? E.g. banded in B-splines?
    Tᵏ + V
end

LinearAlgebra.isposdef(T::Union{Tridiagonal,SymTridiagonal}) = isposdef(Matrix(T))

"""
    PoissonProblem(k, u, v[; w′=similar(u), Tᵏ=get_double_laplacian(R,k)])

Create the Poisson problem of order `k` for the mutual density `u†(r)
.* v(r)`. `w′` is a `MulQuasiVector` of the same kind as `u` and `v`,
and may optionally be provided as a pre-allocated vector, in case it
is e.g. the diagonal of a potential matrix. The same way, the
Laplacian may be reused from other Poisson problems of the same order,
but with different orbitals `u` and/or `v`.
"""
function PoissonProblem(k::Int, u::RO₁, v::RO₁;
                        w′::RO₂=similar(u),
                        Tᵏ::M = get_double_laplacian(u.mul.factors[1],k),
                        kwargs...) where {T,B<:AbstractQuasiMatrix,
                                          RO₁<:RadialOrbital{T,B},
                                          RO₂<:RadialOrbital{T,B},
                                          M<:AbstractMatrix}
    axes(u) == axes(v) || throw(DimensionMismatch("Incompatible axes"))
    Ru,cu = u.mul.factors
    Rv,cv = v.mul.factors
    Ru == Rv || throw(DimensionMismatch("Incompatible bases"))

    ρ = similar(u)

    Tᵏ⁻¹ = LaplacianFactorization(Tᵏ, ρ, w′; kwargs...)

    R = w′.mul.factors[1]
    r = locs(R)

    # # This is how we'd ideally write it, to be basis-agnostic:
    # rₘₐₓ = righendpoint(axes(R,1))
    Δr = r[end]-r[end-1]
    rₘₐₓ = r[end] + Δr

    PoissonProblem(k, inv.(r),
                   R*(r.^k), r.^(k+1), inv(rₘₐₓ^(2k+1)),
                   Tᵏ, u .⋆ v, ρ, w′, Tᵏ⁻¹)
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

- Fischer, C. F., & Guo, W. (1990). Spline algorithms for the
  Hartree-Fock equation for the helium ground state. Journal of
  Computational Physics, 90(2),
  486–496. http://dx.doi.org/10.1016/0021-9991(90)90176-2

- McCurdy, C. W., Baertschy, M., & Rescigno, T. N. (2004). Solving the
  three-body Coulomb breakup problem using exterior complex
  scaling. Journal of Physics B: Atomic, Molecular and Optical
  Physics, 37(17),
  137–187. http://dx.doi.org/10.1088/0953-4075/37/17/r01

=#

function (pp::PoissonProblem)(lazy_density=pp.uv; verbosity=0, io::IO=stdout, kwargs...)
    k,ρ,r⁻¹ = pp.k,pp.ρ,pp.r⁻¹
    ρc = ρ.mul.factors[2]
    wc = pp.w′.mul.factors[2]

    copyto!(ρ, lazy_density) # Form density
    pp.Tᵏ⁻¹.rhs .= (2k+1) * ρc .* r⁻¹
    ldiv!(pp.Tᵏ⁻¹, pp.w′; io=io, kwargs...)

    # Add in homogeneous contribution
    s = (ρ'pp.rᵏ)[1]*pp.rₘₐₓ⁻²ᵏ⁺¹
    wc .+= s*pp.rᵏ⁺¹

    wc .*= r⁻¹

    pp.w′
end

# For exchange potentials, where the density is formed in part from
# the orbital which is "acted upon".
function (pp::PoissonProblem)(v::RO; kwargs...) where {RO<:RadialOrbital}
    R,vc = v.mul.factors
    pp.uv.R == R ||
        throw(DimensionMismatch("Cannot form mutual density from different bases"))
    pp((R*pp.uv.u) .⋆ v; kwargs...)
end

export PoissonProblem

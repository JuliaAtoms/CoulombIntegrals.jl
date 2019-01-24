using FEDVRQuasi
using FiniteDifferencesQuasi
using LinearAlgebra
using IterativeSolvers
import ContinuumArrays: axes
import ContinuumArrays.QuasiArrays: AbstractQuasiMatrix, MulQuasiArray

#=
This method computes the integral \(Y^k(n\ell,n'\ell;r)/r\) where
\[\begin{aligned}
Y^k(n\ell,n'\ell;r)&\equiv
r\int_0^\infty
U^k(r,s)
P(n\ell;s)
P(n'\ell';s)
\mathrm{d}s\\
&=
\int_0^r
\left(\frac{s}{r}\right)^k
P(n\ell;s)
P(n'\ell';s)
\mathrm{d}s+
\int_r^\infty
\left(\frac{r}{s}\right)^{k+1}
P(n\ell;s)
P(n'\ell';s)
\mathrm{d}s
\end{aligned}\]
through the solution of Poisson's problem
\[\left[\frac{\mathrm{d}^2}{\mathrm{d}r^2} -
 \frac{k(k+1)}{r^2}\right]
Y^k(n\ell,n'\ell;r)
=-\frac{2k+1}{r}
P(n\ell;r)P(n'\ell';r)\]
subject to the boundary conditions
\[\begin{cases}
Y^k(n\ell,n'\ell';0) &= 0,\\
\frac{\mathrm{d}}{\mathrm{d}r}
Y^k(n\ell,n'\ell';r) &=
-\frac{k}{r}
Y^k(n\ell,n'\ell';r),\quad
r \to \infty.
\end{cases}\]

=#

function poisson!(w::MulQuasiArray{T,N,<:Mul{<:Tuple,<:Tuple{<:B,<:A}}},
                  ρ::MulQuasiArray{T,N,<:Mul{<:Tuple,<:Tuple{<:B,<:A}}},
                  k::I; kwargs...) where {T,N,B<:AbstractQuasiMatrix{T},A<:AbstractArray,I<:Integer}
    axes(w) == axes(ρ) || throw(DimensionMismatch("Incompatible axes"))
    R,ρc = ρ.mul.factors
    Rw,wc = w.mul.factors

    R == Rw || throw(DimensionMismatch("Incompatible bases"))

    D = Derivative(axes(R,1))
    Tm = R'D'D*R
    r = locs(R)
    Tm -= Diagonal(k*(k+1)./r.^2)

    v = -(2k+1)*ρc./r

    res=cg!(wc, Tm, v; kwargs...)
    :log in keys(kwargs) && display(last(res))

    # Homogeneous contribution
    rk = R*(r.^k)
    s = (ρ'rk)[1]/(r[end]^(2k+1))
    wc .+= s*r.^(k+1)

    wc ./= r

    w
end

const RadialOrbital{T,R} = MulQuasiArray{T,1,<:Mul{<:Tuple, <:Tuple{R,<:AbstractVector}}}

struct SlaterPotential{T,R<:AbstractQuasiMatrix, LD,
                       RO₁<:RadialOrbital{T,R},RO₂<:RadialOrbital{T,R}}
    k::Int
    uv::LD
    ρ::RO₁
    w′::RO₂
    statevars::CGStateVariables{T,Vector{T}}
end

function SlaterPotential(k::Int, u::RO₁, v::RO₁, w′::RO₂=similar(u);
                         kwargs...) where {T,R<:AbstractQuasiMatrix,
                                           RO₁<:RadialOrbital{T,R},
                                           RO₂<:RadialOrbital{T,R}}
    ρ = similar(u)
    w′c = w′.mul.factors[2]
    w′c .= 0
    statevars = CGStateVariables(zero(w′c), similar(w′c), similar(w′c))
    sp = SlaterPotential(k, u .⋆ v, ρ, w′, statevars)
    sp(true; kwargs...)
    sp
end

function (sp::SlaterPotential)(initially_zero=false; kwargs...)
    copyto!(sp.ρ, sp.uv) # Form density
    poisson!(sp.w′, sp.ρ, sp.k;
             statevars=sp.statevars, initially_zero=initially_zero, kwargs...)
    sp.w′
end

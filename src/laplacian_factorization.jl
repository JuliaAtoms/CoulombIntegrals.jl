mutable struct LaplacianFactorization{Factorization,V,RO}
    factorization::Factorization
    rhs::V
    y::RO # Intermediate vector
end

# * Solution via Conjugate Gradient

const ConjugateGradient = Union{CGIterable,PCGIterable}

"""
    cg_iterator!(y::RadialOrbital, Tᵏ::M, ρ::RadialOrbital)

Returns a `PCGIterable` for the Laplacian `Tᵏ` in spherical
coordinates and the mutual density `ρ`; the solution will be stored in
`y` and a preconditioner will be generated using the Ruge–Stuben
solver from AlgebraicMultigrid.jl.
"""
function IterativeSolvers.cg_iterator!(y::RO₂, Tᵏ::M, ρ::RO₁;
                                       kwargs...) where {T,B<:AbstractQuasiMatrix,
                                                         M,
                                                         RO₁<:RadialOrbital{T,B},
                                                         RO₂<:RadialOrbital{T,B}}
    # The reason this method is separate from the
    # LaplacianFactorization constructor is so that we can test the
    # conjugate-gradient-based Poisson solver, even for
    # finite-differences, where we'd normally use optimized
    # factorizations for the Laplacian.
    isposdef(Tᵏ) ||
        @warn "Laplacian matrix not positive-definite, conjugate gradient may not converge"

    yc = y.mul.factors[2]
    yc .= 0

    prec = aspreconditioner(ruge_stuben(sparse(Tᵏ)))
    iterable = cg_iterator!(yc, Tᵏ, ρ.mul.factors[2], prec;
                            initially_zero=true, kwargs...)
    iterable.reltol = √(eps(real(eltype(yc))))

    iterable
end

function LaplacianFactorization(Tᵏ::M, ρ::RO₁, w′::RO₂;
                                kwargs...) where {T,B<:AbstractQuasiMatrix,
                                                  M,
                                                  RO₁<:RadialOrbital{T,B},
                                                  RO₂<:RadialOrbital{T,B}}

    y = similar(w′)
    iterable = cg_iterator!(y, Tᵏ, ρ; kwargs...)

    LaplacianFactorization(iterable, iterable.r, y)
end

function reset_cg_iterator!(it::It) where {It<:ConjugateGradient}
    # It is assumed that the new RHS is already copied to it.r
    it.mv_products = 1
    mul!(it.c, it.A, it.x)
    it.r .-= it.c
    it.residual = norm(it.r)
    it.u .= 0
end

function LinearAlgebra.ldiv!(A::LaplacianFactorization{It,V,RO}, x::RO;
                             io::IO=stdout,
                             verbosity=0) where {It<:ConjugateGradient,V,RO}
    iterable = A.factorization

    reset_cg_iterator!(iterable)
    ii = 0
    for (iteration,item) in enumerate(iterable)
        iterable.mv_products += 1
        verbosity > 1 && println(io, "#$iteration: $(iterable.residual)")
        ii += 1
    end
    verbosity > 0 && println(io,
                            "Poisson problem: Converged: ", IterativeSolvers.converged(iterable) ? "yes" : "no",
                            ", #iterations: ", ii, "/", iterable.maxiter,
                            ", residual: ", iterable.residual)

    copyto!(x.mul.factors[2], A.y.mul.factors[2])
    x
end

# * Solution via factorizations
#
# This mode is used for easily factorizable Laplacians, such as those
# arising from finite-differences treatments.
function LaplacianFactorization(Tᵏ::M, ρ::RO₁, w::RO₂) where {T,B<:AbstractFiniteDifferences,
                                                              M,
                                                              RO₁<:RadialOrbital{T,B},
                                                              RO₂<:RadialOrbital{T,B}}
    rhs = similar(ρ.mul.factors[2])
    LaplacianFactorization(factorize(Tᵏ), rhs, nothing)
end

function LinearAlgebra.ldiv!(A::LaplacianFactorization{F,V,Nothing}, x::RO;
                             kwargs...) where {F,V,RO}
    ldiv!(x.mul.factors[2], A.factorization, A.rhs)
    x
end

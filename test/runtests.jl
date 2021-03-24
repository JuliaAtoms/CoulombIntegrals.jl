using CoulombIntegrals
using CompactBases
using FillArrays
using LazyArrays
using LinearAlgebra
using BandedMatrices
using Parameters
using Test

include("exact_poisson.jl")
include("hydrogenic_orbitals.jl")
include("integrals.jl")
include("test_convergence_rates.jl")

@testset "Basic test" begin
    function get_grid(grid_type, rmax, Z; ρ=0.1, ρmax=0.6, α=0.002)
        ρ /= Z

        if grid_type == :fd_uniform
            N = ceil(Int, rmax/ρ)
            FiniteDifferences(N, ρ)
        elseif grid_type == :fd_loglin
            StaggeredFiniteDifferences(0.1ρ, ρmax, α, rmax)
        elseif grid_type == :implicit_fd
            N = ceil(Int, rmax/ρ)
            ImplicitFiniteDifferences(N, ρ)
        elseif grid_type == :fedvr
            t = range(0, stop=rmax, length=41)
            FEDVR(t, Vcat(10,Fill(7,length(t)-2)))[:,2:end-1]
        elseif grid_type == :bsplines
            t = ExpKnotSet(5, -2.0, log10(rmax), 200)
            BSpline(t)[:,2:end-1]
        end
    end

    n,ℓ = 3,0
    n′,ℓ′ = 2,0
    k = abs(ℓ′-ℓ) # Multipole order

    Z = 1.0
    rmax = 40(max(n,n′)^1.05)/Z

    grid_types = [(:fd_uniform, 2e-4, 4e-8),
                  (:fd_loglin, 1e-6, 5e-10),
                  (:implicit_fd, 2e-5, 4e-8),
                  (:fedvr, 3e-7, 8e-11),
                  (:bsplines, 5e-9, 2e-9)]

    # All in the name of type-stability
    Yk_scaled(Yk_f) = (r,Z) -> Yk_f(Z*r)
    Yk_ref = exact_Yᵏs[((n,ℓ),(n′,ℓ′),k)]
    fYk = Base.Fix2(Yk_scaled(Yk_ref), Z)

    ϕf = hydredwfn(n, ℓ, Z)
    ϕ′f = hydredwfn(n′, ℓ′, Z)

    @testset "Grid = $(grid_type)" for (grid_type,tol,asymtol) in grid_types
        R = get_grid(grid_type, rmax, Z, ρ=0.1, ρmax=1.0)
        r = axes(R,1)

        Ỹᵏ = R \ fYk.(r)

        ϕ = R \ ϕf.(r)
        ϕ′ = R \ ϕ′f.(r)

        o = R \ one.(r)
        out1 = similar(o)
        out2 = similar(o)

        ρ = Density(applied(*,R,ϕ), applied(*,R,ϕ′))
        potential = CoulombRepulsionPotential(R, k)
        copyto!(potential, ρ)
        Y = potential.poisson.Y
        @test Y ≈ Ỹᵏ atol=tol

        mpot = Matrix(potential)
        @test mpot isa (R isa CompactBases.BSplineOrRestricted ? BandedMatrix : Diagonal)

        mul!(out1, potential, o)
        mul!(out2, LinearOperator(mpot, R), o)
        if R isa CompactBases.BSplineOrRestricted
            # Matricization does yield the same behaviour for the
            # first few elements since the Dirichlet0 boundary
            # condition at r=0 does not allow the accurate
            # representation of Y/r.
            @test out1[55:end] ≈ out2[55:end] atol=tol
        else
            @test out1 ≈ out2 atol=tol
        end

        @testset "AsymptoticPoissonProblem" begin
            R̃ = R[:,1:end-5]
            apoisson = AsymptoticPoissonProblem(R, k, R̃)
            CoulombIntegrals.solve!(apoisson, ρ)
            @test Y ≈ apoisson.Y atol=asymtol
        end
    end
end

include("poisson_problem_hermiticity.jl")

@testset "Accuracy tests" begin
    rₘₐₓ = 100.0

    orbital_modes = [:symbolic]
    # # Arnoldi is not stable with very small grid spacings;
    # # shift-and-invert has to be implemented.
    # orbital_modes = [:symbolic, :arnoldi]

    coulomb_modes = [:poisson]
    # # Direct integration is simply too slow.
    # coulomb_modes = [:poisson, :direct]
    Z = 1.0

    fedvr_order = 10

    fedvr_inputs(exponents,errors) =
        ("FEDVR",
         (rₘₐₓ,N) -> begin
         t = range(0,stop=rₘₐₓ,length=N)
         amended_order = vcat(fedvr_order+2, Fill(fedvr_order,length(t)-2))
         R = FEDVR(t, amended_order)[:,2:end-1]
         R,(t[2]-t[1])/fedvr_order
         end,
         unique(ceil.(Int, 2 .^ exponents)),
         errors)

    fd_inputs(exponents,errors) =
        ("Finite-differences",
         (rₘₐₓ,N) -> begin
         R = StaggeredFiniteDifferences(rₘₐₓ,N)
         R,step(R)
         end,
         unique(ceil.(Int, 2 .^ exponents)),
         errors)

    # TODO: Add B-spline convergence tests

    @testset "Yᵏ convergence rates" begin
        @testset "$Basis" for (Basis,Rf,Ns,errors) in [
            fedvr_inputs(range(1,stop=5,length=20),
                         [(6, "local", 2e-5, 7.0)]),
            fd_inputs(range(6.5,stop=10,length=15),
                      [(6, "local", 1e-3, 2.0)])
        ]
            @test test_convergence_rates(
                Rf, rₘₐₓ, Ns, 2,
                errors,
                ["N", "ρ", "a", "b", "k", "Local δ", "Global δ", "Time"],
                orbital_modes, coulomb_modes,
                sort(collect(keys(exact_Yᵏs)))) do cache, orbital_mode, coulomb_mode, ((n,ℓ), (n′,ℓ′), k)
                    (n′ > n || n′ == n && ℓ′ > ℓ) && return nothing
                    Yᵏ_error(cache,
                             k, n, ℓ, n′, ℓ′, Z,
                             orbital_mode, coulomb_mode)
                end
        end
    end

    @testset "Fᵏ convergence rates" begin
        @testset "$Basis" for (Basis,Rf,Ns,errors) in [
            fedvr_inputs(range(2,stop=6,length=20),
                         [(11,"δF",2e-6,6.0),
                          (13,"δF′",2e-6,6.0)]),
            fd_inputs(range(7,stop=11,length=15),
                      [(11,"δF",2e-4,2.0),
                       (13,"δF′",2e-4,2.0)])
        ]
            @test test_convergence_rates(
                Rf, rₘₐₓ, Ns, 2,
                errors,
                ["N", "ρ", "a", "b", "c", "d", "k", "Exact", "Exact",
                 "F", "δ", "F′", "δ′", "F-F′", "Time"],
                orbital_modes, coulomb_modes, collect(keys(exact_Fᵏs))) do cache, orbital_mode, coulomb_mode, ((n,ℓ),(n′,ℓ′),k)
                    Fᵏ_error(cache,
                             k, n, ℓ, n′, ℓ′,
                             Z, orbital_mode, coulomb_mode)
                end
        end
    end

    @testset "Gᵏ convergence rates" begin
        @testset "$Basis" for (Basis,Rf,Ns,errors) in [
            fedvr_inputs(range(2,stop=6,length=20),
                         [(11,"δG",2e-6,6.0)]),
            fd_inputs(range(7,stop=12,length=15),
                      [(11,"δG",2e-4,2.0)])
        ]
            @test test_convergence_rates(
                Rf, rₘₐₓ, Ns, 2,
                errors,
                ["N", "ρ", "a", "b", "c", "d", "k", "Exact", "Exact",
                 "G", "δ", "Time"],
                orbital_modes, coulomb_modes, collect(keys(exact_Gᵏs))) do cache, orbital_mode, coulomb_mode, ((n,ℓ),(n′,ℓ′),k)
                    Gᵏ_error(cache,
                             k, n, ℓ, n′, ℓ′,
                             Z, orbital_mode, coulomb_mode)
                end
        end
    end
end

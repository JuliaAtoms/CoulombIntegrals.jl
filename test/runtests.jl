using CoulombIntegrals
using FiniteDifferencesQuasi
using Test

include("exact_poisson.jl")
include("hydrogenic_orbitals.jl")
include("integrals.jl")
include("test_convergence_rates.jl")

@testset "Accuracy tests" begin
    rₘₐₓ = 100

    orbital_modes = [:symbolic]
    # # Arnoldi is not stable with very small grid spacings;
    # # shift-and-invert has to be implemented.
    # orbital_modes = [:symbolic, :arnoldi]

    coulomb_modes = [:poisson]
    # # Direct integration is simply too slow.
    # coulomb_modes = [:poisson, :direct]
    Zs = 1:3

    Rs = [(rₘₐₓ,N,Z) -> begin
          R = RadialDifferences(rₘₐₓ,N,Z)
          R,step(R)
          end]


    @testset "Yᵏ convergence rates" begin
        Ns = 2 .^ (8:13)
        @test test_convergence_rates(
            Ns, 2,
            [(6, "local", 2e-5, 2.0), # Local error should decrease quadratically
             (7, "global", 4e-2, 1.0)], # Global error should decrease linearly
            ["N", "ρ", "a", "b", "k", "Local δ", "Global δ", "Time"],
            orbital_modes, coulomb_modes, Rs, Zs, collect(keys(exact_Yᵏs))) do N, orbital_mode, coulomb_mode, R, Z, ((n,ℓ), (n′,ℓ′), k)
                (n′ > n || n′ == n && ℓ′ > ℓ) && return nothing
                Yᵏ_error(R(rₘₐₓ/Z, N, Z)...,
                         k, n, ℓ, n′, ℓ′, Z,
                         orbital_mode, coulomb_mode)
            end
    end

    @testset "Fᵏ convergence rates" begin
        Ns = 2 .^ (8:13)
        @test test_convergence_rates(
            Ns, 2,
            [(11,"δF",3e-5,2.0),
             (13,"δF′",3e-5,2.0)],
            ["N", "ρ", "a", "b", "c", "d", "k", "Exact", "Exact",
             "F", "δ", "F′", "δ′", "F-F′", "Time"],
            orbital_modes, coulomb_modes, Rs, Zs, collect(keys(exact_Fᵏs))) do N, orbital_mode, coulomb_mode, R, Z, ((n,ℓ),(n′,ℓ′),k)
                Fᵏ_error(R(rₘₐₓ/Z, N, Z)...,
                         k, n, ℓ, n′, ℓ′,
                         Z, orbital_mode, coulomb_mode)
            end
    end

    @testset "Gᵏ convergence rates" begin
        Ns = 2 .^ (9:13)
        @test test_convergence_rates(
            Ns, 2,
            [(11,"δG",3e-5,2.0)],
            ["N", "ρ", "a", "b", "c", "d", "k", "Exact", "Exact",
             "G", "δ", "Time"],
            orbital_modes, coulomb_modes, Rs, Zs, collect(keys(exact_Gᵏs))) do N, orbital_mode, coulomb_mode, R, Z, ((n,ℓ),(n′,ℓ′),k)
                Gᵏ_error(R(rₘₐₓ/Z, N, Z)...,
                         k, n, ℓ, n′, ℓ′,
                         Z, orbital_mode, coulomb_mode)
            end
    end
end

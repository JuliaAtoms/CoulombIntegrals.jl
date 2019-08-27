using CoulombIntegrals
using FEDVRQuasi
using FillArrays
using LazyArrays
using FiniteDifferencesQuasi
using Test

include("exact_poisson.jl")
include("hydrogenic_orbitals.jl")
include("integrals.jl")
include("test_convergence_rates.jl")

@testset "Accuracy tests" begin
    rₘₐₓ = 100.0

    orbital_modes = [:symbolic]
    # # Arnoldi is not stable with very small grid spacings;
    # # shift-and-invert has to be implemented.
    # orbital_modes = [:symbolic, :arnoldi]

    coulomb_modes = [:poisson]
    # # Direct integration is simply too slow.
    # coulomb_modes = [:poisson, :direct]
    Z = 1

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
         R = RadialDifferences(rₘₐₓ,N,Z)
         R,step(R)
         end,
         unique(ceil.(Int, 2 .^ exponents)),
         errors)

    @testset "Yᵏ convergence rates" begin
        @testset "$Basis" for (Basis,Rf,Ns,errors) in [
            fedvr_inputs(range(1,stop=5,length=30),
                         [(6, "local", 2e-5, 7.0)]),
            fd_inputs(range(6.5,stop=10,length=15),
                      [(6, "local", 1e-3, 2.0)])
        ]
            @test test_convergence_rates(
                Rf, rₘₐₓ, Ns, 2,
                errors,
                ["N", "ρ", "a", "b", "k", "Local δ", "Global δ", "Time"],
                orbital_modes, coulomb_modes,
                sort(collect(keys(exact_Yᵏs)))) do R, ρ, V, poissons, orbital_mode, coulomb_mode, ((n,ℓ), (n′,ℓ′), k)
                    (n′ > n || n′ == n && ℓ′ > ℓ) && return nothing
                    Yᵏ_error(R, ρ,
                             k, n, ℓ, n′, ℓ′, Z,
                             orbital_mode, coulomb_mode, V, poissons)
                end
        end
    end

    @testset "Fᵏ convergence rates" begin
        @testset "$Basis" for (Basis,Rf,Ns,errors) in [
            fedvr_inputs(range(2,stop=6,length=30),
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
                orbital_modes, coulomb_modes, collect(keys(exact_Fᵏs))) do R, ρ, V, poissons, orbital_mode, coulomb_mode, ((n,ℓ),(n′,ℓ′),k)
                    Fᵏ_error(R, ρ,
                             k, n, ℓ, n′, ℓ′,
                             Z, orbital_mode, coulomb_mode, V, poissons)
                end
        end
    end

    @testset "Gᵏ convergence rates" begin
        @testset "$Basis" for (Basis,Rf,Ns,errors) in [
            fedvr_inputs(range(2,stop=6,length=30),
                         [(11,"δG",2e-6,6.0)]),
            fd_inputs(range(7,stop=12,length=15),
                      [(11,"δG",2e-4,2.0)])
        ]
            @test test_convergence_rates(
                Rf, rₘₐₓ, Ns, 2,
                errors,
                ["N", "ρ", "a", "b", "c", "d", "k", "Exact", "Exact",
                 "G", "δ", "Time"],
                orbital_modes, coulomb_modes, collect(keys(exact_Gᵏs))) do R, ρ, V, poissons, orbital_mode, coulomb_mode, ((n,ℓ),(n′,ℓ′),k)
                    Gᵏ_error(R, ρ,
                             k, n, ℓ, n′, ℓ′,
                             Z, orbital_mode, coulomb_mode, V, poissons)
                end
        end
    end
end

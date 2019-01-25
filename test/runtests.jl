using CoulombIntegrals
using FiniteDifferencesQuasi
using Test

include("exact_poisson.jl")
include("test_poisson.jl")

# These accuracies are not fantastic, but they ensure that no
# regressions occur.
function F_abstol(::RadialDifferences, a, b)
    if a == b == (1,0)
        0.03
    elseif a < (4,0)
        0.007
    elseif a == b == (4,0)
        2e-4
    else
        1e-4
    end
end

@testset "Radial differences" begin
    rₘₐₓ = 300

    @testset "Hydrogen" begin
        ρ = 0.25
        N = ceil(Int, rₘₐₓ/ρ + 1/2)
        test_poisson(RadialDifferences(N, ρ, 1), 1, 0.05, F_abstol)
    end
    @testset "Helium" begin
        ρ = 0.2
        N = ceil(Int, rₘₐₓ/ρ + 1/2)
        test_poisson(RadialDifferences(N, ρ, 2), 2, 0.08, F_abstol)
    end
end

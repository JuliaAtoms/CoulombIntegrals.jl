function exchange_action!(w, potential, u, v, x, ϕ, R)
    z = exp(im*ϕ)
    ρ = Density(u, applied(*, R, z*x))
    copyto!(potential, ρ)
    mul!(w, potential, v.args[2], 1/z)
end

function exchange_action(potential, u, v, ϕ, R)
    n = size(R,2)
    eye = complex(Matrix(1.0I, n, n))
    out = similar(eye)
    for j = 1:size(eye,2)
        exchange_action!(view(out, :, j),
                         potential, u, v,
                         view(eye, :, j), ϕ, R)
    end
    out
end

@testset "Poisson problem Hermiticity" begin
    N = 100
    ρ = 0.2

    R = StaggeredFiniteDifferences(N, ρ, 0.0)
    r = axes(R,1)
    D = Derivative(r)
    ∇² = apply(*, R', D', D, R)

    T = ∇²/-2
    V = apply(*, R', QuasiDiagonal(-1 ./ r), R)
    H = T+V

    ee = eigen(H)

    @testset "n = $(n)" for n = 1:5
        potential = CoulombRepulsionPotential(R, 0, ComplexF64)
        u = applied(*, R, complex(ee.vectors[:,n]))
        v = applied(*, R, complex(ee.vectors[:,n]))

        @testset "ϕ = $(ϕ)" for ϕ ∈ π*(0:1/8:2)
            A = exchange_action(potential, u, v, ϕ, R)
            @test norm(A-A') < 1e-16
        end
    end
end

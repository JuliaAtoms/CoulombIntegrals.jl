function exchange_action!(w, poisson, V, v, ϕ, R)
    z = exp(im*ϕ)
    poisson(applied(*, R, z*v))
    mul!(w, V, poisson.uv.v, 1/z, false)
    w
end

function exchange_action(poisson, V, ϕ)
    eye = complex(Matrix(1.0I, size(V)))
    out = similar(eye)
    R = poisson.ρ.args[1]
    for j = 1:size(eye,2)
        exchange_action!(view(out, :, j), poisson, V,
                         view(eye, :, j), ϕ, R)
    end
    out
end

@testset "Poisson problem Hermiticity" begin
    N = 100
    ρ = 0.2

    R = RadialDifferences(N, ρ)
    r = axes(R,1)
    D = Derivative(r)
    ∇² = apply(*, R', D', D, R)

    T = ∇²/-2
    V = apply(*, R', QuasiDiagonal(-1 ./ r), R)
    H = T+V

    ee = eigen(H)

    @testset "n = $(n)" for n = 1:5
        u = applied(*, R, complex(ee.vectors[:,n]))
        v = applied(*, R, complex(ee.vectors[:,n]))
        w = similar(u)
        V = Diagonal(w.args[2])
        poisson = PoissonProblem(0, u, v, w′ = w)

        @testset "ϕ = $(ϕ)" for ϕ ∈ π*(0:1/8:2)
            A = exchange_action(poisson, V, ϕ)
            @test norm(A-A') < 1e-16
        end
    end
end

using LinearAlgebra
using ArnoldiMethod
using PrettyTables
import ContinuumArrays.QuasiArrays: AbstractQuasiMatrix

function orb_label((n,ℓ))
    ℓs = "spdfg"[ℓ+1]
    "$n$(ℓs)"
end

function get_orbitals(R::B, ℓ::Int, Z, nev::Int) where {B<:AbstractQuasiMatrix}
    println("Finding eigenstates for Z = $Z, ℓ = $ℓ")
    Tℓ = CoulombIntegrals.get_double_laplacian(R,ℓ)
    Tℓ ./= 2
    V = Matrix(r -> -Z/r, R)
    H = Tℓ + V

    schur,history = partialschur(H, nev=nev, tol=sqrt(eps()), which=SR())
    println(history)
    println(diag(schur.R))

    [(n+ℓ,ℓ) => normalize!(R*schur.Q[:,n]) for n = 1:nev]
end

function test_poisson(R::B, Z, Yᵏ_reltol, F_abstol::Function) where {B<:AbstractQuasiMatrix}
    r = CoulombIntegrals.locs(R)

    radial_orbitals = Dict(vcat([get_orbitals(R, ℓ, Z, 4-ℓ) for ℓ ∈ 0:3]...))
    
    println("Testing various Yᵏ:s")
    Ydata = map(collect(keys(exact_Yᵏs))) do (orb,k)
        exact = exact_Yᵏs[(orb,k)]
        Π = PoissonProblem(k, radial_orbitals[orb], radial_orbitals[orb])
        Π()
        @test Π.solve_iterable.mv_products < 10
        Π()
        @test Π.solve_iterable.mv_products == 1
        δ = R*((R'Π.w′).*r - exact.(Z*r))
        nδ = norm(δ)
        [orb orb_label(orb) k nδ nδ/norm(Π.w′)]
    end |> d -> sortslices(vcat(d...), dims=1)
    pretty_table(Ydata[:,2:end], ["o", "k", "δ", "relδ"])
    @test all(Ydata[:,end] .< Yᵏ_reltol)

    println("Testing various F⁰s")
    Fdata = map(collect(keys(exact_F⁰s))) do (a,b,k)
        va,vb = radial_orbitals[a], radial_orbitals[b]
        # The F integrals are symmetric, and thus which orbital is used to
        # form the Yᵏ potential should not matter.
        Π = PoissonProblem(k, va, va)
        Π()
        @test Π.solve_iterable.mv_products < 10
        Π()
        @test Π.solve_iterable.mv_products == 1
        Π′ = PoissonProblem(k, vb, vb)
        Π′()
        @test Π′.solve_iterable.mv_products < 10
        F = (vb.*vb)'Π.w′
        F′ = (va.*va)'Π′.w′
        ev = exact_F⁰s[(a,b,k)]*Z
        evf = convert(Float64, ev)
        [a b orb_label(a) orb_label(b) k ev evf F F-evf F′ F′-evf F-F′]
    end |> d -> sortslices(vcat(d...), dims=1)
    pretty_table(Fdata[:,3:end], ["a", "b", "k", "Exact", "Exact", "F", "δ", "F′", "δ′", "F-F′"])
    @test all(Fdata[:,8] .≈ Fdata[:,10])
    for i in 1:size(Fdata,1)
        @test isapprox(Fdata[i,7], Fdata[i,8], atol=F_abstol(R,Fdata[i,1],Fdata[i,2]))
    end

    println("Testing various Gᵏs")
    Gdata = map(collect(keys(exact_Gᵏs))) do (a,b,k)
        va,vb = radial_orbitals[a], radial_orbitals[b]
        Π = PoissonProblem(k, va, vb)
        Π()
        @test Π.solve_iterable.mv_products < 10
        Π()
        @test Π.solve_iterable.mv_products == 1
        G = (va.*vb)'Π.w′
        ev = exact_Gᵏs[(a,b,k)]*Z
        evf = convert(Float64, ev)
        [a b orb_label(a) orb_label(b) k ev evf G G-evf]
    end |> d -> sortslices(vcat(d...), dims=1)
    pretty_table(Gdata[:,3:end], ["a", "b", "k", "Exact", "Exact", "G", "δ"])
    @test all(isapprox.(Gdata[:,7],Gdata[:,8],atol=1e-3))
end

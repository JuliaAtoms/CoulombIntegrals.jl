function get_Yᵏ(R,k,n,ℓ,n′,ℓ′,Z,orbital_mode,coulomb_mode,V,poissons; kwargs...)
    u = get_orbitals(R, ℓ, Z, n-ℓ, orbital_mode,V)[end][2]
    v = get_orbitals(R, ℓ′, Z, n′-ℓ′, orbital_mode,V)[end][2]
    r = CoulombIntegrals.locs(R)
    if coulomb_mode ∈ [:poisson,:poisson_cg]
        # TODO: Enforce conjugate-gradient when `:poisson_cg`
        Π = if k ∈ keys(poissons)
            poissons[k]
        else
            poissons[k] = PoissonProblem(k, u, v, kwargs...)
        end
        Π.y.args[2] .= false
        Π(u .⋆ v, verbosity=0)
        if coulomb_mode == :poisson_cg
            @test Π.solve_iterable.mv_products < 10
            Π()
            @test Π.solve_iterable.mv_products == 1
        end
        Π.w′.args[2].*r
    elseif coulomb_mode == :direct
        LCk = LazyCoulomb(R,k)
        M = materialize(applied(*,u',LCk,v))
        w = M .* r
    else
        throw(ArgumentError("Unknown coulomb_mode $(coulomb_mode)"))
    end
end

function get_Fᵏ(R,k,n,ℓ,n′,ℓ′,Z,orbital_mode,coulomb_mode,V,poissons)
    u = get_orbitals(R, ℓ, Z, n-ℓ, orbital_mode,V)[end][2]
    v = get_orbitals(R, ℓ′, Z, n′-ℓ′, orbital_mode,V)[end][2]
    r = CoulombIntegrals.locs(R)

    # The F integrals are symmetric, and thus which orbital is used to
    # form the Yᵏ potential should not matter.
    Yᵏ = get_Yᵏ(R,k,n,ℓ,n,ℓ,Z,orbital_mode,coulomb_mode,V,poissons)
    Yᵏ′ = get_Yᵏ(R,k,n′,ℓ′,n′,ℓ′,Z,orbital_mode,coulomb_mode,V,poissons)

    uu = materialize(u)
    vv = materialize(v)

    # Perform the integral over the other coordinate.
    Fᵏ = (vv.*vv)'*(R*(Yᵏ./r))
    Fᵏ′ = (uu.*uu)'*(R*(Yᵏ′./r))

    Fᵏ,Fᵏ′
end

function get_Gᵏ(R,k,n,ℓ,n′,ℓ′,Z,orbital_mode,coulomb_mode,V,poissons)
    u = get_orbitals(R, ℓ, Z, n-ℓ, orbital_mode,V)[end][2]
    v = get_orbitals(R, ℓ′, Z, n′-ℓ′, orbital_mode,V)[end][2]
    r = CoulombIntegrals.locs(R)

    Yᵏ = get_Yᵏ(R,k,n,ℓ,n′,ℓ′,Z,orbital_mode,coulomb_mode,V,poissons)

    uu = materialize(u)
    vv = materialize(v)

    Gᵏ = (uu.*vv)'*(R*(Yᵏ./r))
end

function Yᵏ_error(R, ρ, k, n, ℓ, n′, ℓ′, Z, orbital_mode, coulomb_mode,V,poissons)
    N = size(R,2)
    r = CoulombIntegrals.locs(R)

    t = time()
    Yᵏ = get_Yᵏ(R,k,n,ℓ,n′,ℓ′,Z,orbital_mode,coulomb_mode,V,poissons)
    el = time()-t
    Ỹᵏ = V \ exact_Yᵏs[((n,ℓ),(n′,ℓ′),k)].(Z*r)
    # Ugly work-around
    CoulombIntegrals.weightit!(applied(*, R, Ỹᵏ))

    dYᵏ = Yᵏ - Ỹᵏ
    lδ = maximum(abs, dYᵏ)
    # The way we estimate the global error, by simply summing the
    # magnitudes of the local errors, decreases the order of
    # convergence by 1.
    gδ = sum(abs, dYᵏ)

    a = (n,ℓ)
    b = (n′,ℓ′)

    [N ρ orb_label(a) orb_label(b) k lδ gδ el]
end

function Fᵏ_error(R, ρ, k, n, ℓ, n′, ℓ′, Z, orbital_mode, coulomb_mode,V,poissons)
    N = size(R,2)
    r = CoulombIntegrals.locs(R)

    t = time()
    Fᵏ,Fᵏ′ = get_Fᵏ(R,k,n,ℓ,n′,ℓ′,Z,orbital_mode,coulomb_mode,V,poissons)
    el = time()-t

    a = (n,ℓ)
    b = (n′,ℓ′)
    ev = exact_Fᵏs[(a,b,k)]*Z
    evf = convert(Float64, ev)
    [N ρ orb_label(a) orb_label(b) orb_label(a) orb_label(b) k ev evf Fᵏ Fᵏ-evf Fᵏ′ Fᵏ′-evf Fᵏ-Fᵏ′ el]
end

function Gᵏ_error(R, ρ, k, n, ℓ, n′, ℓ′, Z, orbital_mode, coulomb_mode,V,poissons)
    N = size(R,2)
    r = CoulombIntegrals.locs(R)

    t = time()
    Gᵏ = get_Gᵏ(R,k,n,ℓ,n′,ℓ′,Z,orbital_mode,coulomb_mode,V,poissons)
    el = time()-t

    a = (n,ℓ)
    b = (n′,ℓ′)
    ev = exact_Gᵏs[(a,b,k)]*Z
    evf = convert(Float64, ev)
    [N ρ orb_label(a) orb_label(b) orb_label(b) orb_label(a) k ev evf Gᵏ Gᵏ-evf el]
end

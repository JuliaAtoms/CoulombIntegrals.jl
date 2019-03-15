function get_Yᵏ(R,k,n,ℓ,n′,ℓ′,Z,orbital_mode,coulomb_mode; kwargs...)
    u = get_orbitals(R, ℓ, Z, n-ℓ, orbital_mode)[end][2]
    v = get_orbitals(R, ℓ′, Z, n′-ℓ′, orbital_mode)[end][2]
    r = CoulombIntegrals.locs(R)
    if coulomb_mode ∈ [:poisson,:poisson_cg]
        # TODO: Enforce conjugate-gradient when `:poisson_cg`
        Π = PoissonProblem(k, u, v, kwargs...)
        Π(verbosity=0)
        if coulomb_mode == :poisson_cg
            @test Π.solve_iterable.mv_products < 10
            Π()
            @test Π.solve_iterable.mv_products == 1
        end
        (R'Π.w′).*r
    elseif coulomb_mode == :direct
        LCk = LazyCoulomb(R,k)
        M = materialize(applied(*,u',LCk,v))
        w = M .* r
    else
        throw(ArgumentError("Unknown coulomb_mode $(coulomb_mode)"))
    end
end

function get_Fᵏ(R,k,n,ℓ,n′,ℓ′,Z,orbital_mode,coulomb_mode)
    u = get_orbitals(R, ℓ, Z, n-ℓ, orbital_mode)[end][2]
    v = get_orbitals(R, ℓ′, Z, n′-ℓ′, orbital_mode)[end][2]
    r = CoulombIntegrals.locs(R)
    
    # The F integrals are symmetric, and thus which orbital is used to
    # form the Yᵏ potential should not matter.
    Yᵏ = get_Yᵏ(R,k,n,ℓ,n,ℓ,Z,orbital_mode,coulomb_mode)
    Yᵏ′ = get_Yᵏ(R,k,n′,ℓ′,n′,ℓ′,Z,orbital_mode,coulomb_mode)

    # Perform the integral over the other coordinate.
    Fᵏ = (v.*v)'*R*(Yᵏ./r)
    Fᵏ′ = (u.*u)'*R*(Yᵏ′./r)
    
    Fᵏ,Fᵏ′
end

function get_Gᵏ(R,k,n,ℓ,n′,ℓ′,Z,orbital_mode,coulomb_mode)
    u = get_orbitals(R, ℓ, Z, n-ℓ, orbital_mode)[end][2]
    v = get_orbitals(R, ℓ′, Z, n′-ℓ′, orbital_mode)[end][2]
    r = CoulombIntegrals.locs(R)
    
    Yᵏ = get_Yᵏ(R,k,n,ℓ,n′,ℓ′,Z,orbital_mode,coulomb_mode)
    
    Gᵏ = (u.*v)'*R*(Yᵏ./r)
end

function Yᵏ_error(R, ρ, k, n, ℓ, n′, ℓ′, Z, orbital_mode, coulomb_mode)
    N = size(R,2)
    r = CoulombIntegrals.locs(R)

    t = time()
    Yᵏ = get_Yᵏ(R,k,n,ℓ,n′,ℓ′,Z,orbital_mode,coulomb_mode)
    el = time()-t
    Ỹᵏ = exact_Yᵏs[((n,ℓ),(n′,ℓ′),k)].(Z*r)

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

function Fᵏ_error(R, ρ, k, n, ℓ, n′, ℓ′, Z, orbital_mode, coulomb_mode)
    N = size(R,2)
    r = CoulombIntegrals.locs(R)

    t = time()
    Fᵏ,Fᵏ′ = get_Fᵏ(R,k,n,ℓ,n′,ℓ′,Z,orbital_mode,coulomb_mode)
    el = time()-t

    a = (n,ℓ)
    b = (n′,ℓ′)
    ev = exact_Fᵏs[(a,b,k)]*Z
    evf = convert(Float64, ev)
    [N ρ orb_label(a) orb_label(b) orb_label(a) orb_label(b) k ev evf Fᵏ Fᵏ-evf Fᵏ′ Fᵏ′-evf Fᵏ-Fᵏ′ el]
end

function Gᵏ_error(R, ρ, k, n, ℓ, n′, ℓ′, Z, orbital_mode, coulomb_mode)
    N = size(R,2)
    r = CoulombIntegrals.locs(R)

    t = time()
    Gᵏ = get_Gᵏ(R,k,n,ℓ,n′,ℓ′,Z,orbital_mode,coulomb_mode)
    el = time()-t

    a = (n,ℓ)
    b = (n′,ℓ′)
    ev = exact_Gᵏs[(a,b,k)]*Z
    evf = convert(Float64, ev)
    [N ρ orb_label(a) orb_label(b) orb_label(b) orb_label(a) k ev evf Gᵏ Gᵏ-evf el]
end

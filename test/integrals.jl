function get_Yᵏ(R,k,n,ℓ,n′,ℓ′,Z,orbital_mode,coulomb_mode,V,poissons,args...)
    u = get_orbitals(R, ℓ, Z, n-ℓ, orbital_mode)[end][2]
    v = get_orbitals(R, ℓ′, Z, n′-ℓ′, orbital_mode)[end][2]
    r = CoulombIntegrals.locs(R)
    if coulomb_mode == :poisson
        Π = if k ∈ keys(poissons)
            poissons[k]
        else
            poissons[k] = PoissonProblem(R, k)
        end
        Π.Y .= false
        ρ = Density(u, v)
        solve!(Π, ρ)
        copy(Π.Y)
    elseif coulomb_mode == :direct
        LCk = LazyCoulomb(R,k)
        M = materialize(applied(*,u',LCk,v))
        w = M .* r
    else
        throw(ArgumentError("Unknown coulomb_mode $(coulomb_mode)"))
    end
end

function get_Fᵏ(R,k,n,ℓ,n′,ℓ′,Z,orbital_mode,coulomb_mode, V, S, r⁻¹,poissons,args...)
    u = get_orbitals(R, ℓ, Z, n-ℓ, orbital_mode)[end][2]
    v = get_orbitals(R, ℓ′, Z, n′-ℓ′, orbital_mode)[end][2]
    r = CoulombIntegrals.locs(R)

    # The F integrals are symmetric, and thus which orbital is used to
    # form the Yᵏ potential should not matter.
    Yᵏ = get_Yᵏ(R,k,n,ℓ,n,ℓ,Z,orbital_mode,coulomb_mode,V,poissons,args...)
    Yᵏ′ = get_Yᵏ(R,k,n′,ℓ′,n′,ℓ′,Z,orbital_mode,coulomb_mode,V,poissons,args...)

    ρv = Density(v, v)
    ρu = Density(u, u)

    # Perform the integral over the other coordinate.
    Fᵏ = dot(ρv.ρ, S, r⁻¹*Yᵏ)
    Fᵏ′ = dot(ρu.ρ, S, r⁻¹*Yᵏ′)

    Fᵏ,Fᵏ′
end

function get_Gᵏ(R,k,n,ℓ,n′,ℓ′,Z,orbital_mode,coulomb_mode, V, S, r⁻¹,poissons,args...)
    u = get_orbitals(R, ℓ, Z, n-ℓ, orbital_mode)[end][2]
    v = get_orbitals(R, ℓ′, Z, n′-ℓ′, orbital_mode)[end][2]
    r = CoulombIntegrals.locs(R)

    Yᵏ = get_Yᵏ(R,k,n,ℓ,n′,ℓ′,Z,orbital_mode,coulomb_mode,V,poissons,args...)

    ρ = Density(u, v)
    
    # Perform the integral over the other coordinate.
    Gᵏ = dot(ρ.ρ, S, r⁻¹*Yᵏ)
end

function Yᵏ_error(cache, k, n, ℓ, n′, ℓ′, Z, orbital_mode, coulomb_mode, args...)
    @unpack R,ρ,V,poissons = cache
    N = size(R,2)
    # r = CoulombIntegrals.locs(R)
    r = axes(R,1)

    t = time()
    Yᵏ = get_Yᵏ(R,k,n,ℓ,n′,ℓ′,Z,orbital_mode,coulomb_mode,V,poissons,args...)
    el = time()-t
    Ỹᵏ = R \ exact_Yᵏs[((n,ℓ),(n′,ℓ′),k)].(Z*r)

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

function Fᵏ_error(cache, k, n, ℓ, n′, ℓ′, Z, orbital_mode, coulomb_mode, args...)
    @unpack R, ρ, V, S, r⁻¹, poissons = cache
    N = size(R,2)
    r = CoulombIntegrals.locs(R)

    t = time()
    Fᵏ,Fᵏ′ = get_Fᵏ(R,k,n,ℓ,n′,ℓ′,Z,orbital_mode,coulomb_mode, V, S, r⁻¹,poissons,args...)
    el = time()-t

    a = (n,ℓ)
    b = (n′,ℓ′)
    ev = exact_Fᵏs[(a,b,k)]*Z
    evf = convert(Float64, ev)
    [N ρ orb_label(a) orb_label(b) orb_label(a) orb_label(b) k ev evf Fᵏ Fᵏ-evf Fᵏ′ Fᵏ′-evf Fᵏ-Fᵏ′ el]
end

function Gᵏ_error(cache, k, n, ℓ, n′, ℓ′, Z, orbital_mode, coulomb_mode, args...)
    @unpack R, ρ, V, S, r⁻¹, poissons = cache
    N = size(R,2)
    r = CoulombIntegrals.locs(R)

    t = time()
    Gᵏ = get_Gᵏ(R,k,n,ℓ,n′,ℓ′,Z,orbital_mode,coulomb_mode, V, S, r⁻¹,poissons,args...)
    el = time()-t

    a = (n,ℓ)
    b = (n′,ℓ′)
    ev = exact_Gᵏs[(a,b,k)]*Z
    evf = convert(Float64, ev)
    [N ρ orb_label(a) orb_label(b) orb_label(b) orb_label(a) k ev evf Gᵏ Gᵏ-evf el]
end

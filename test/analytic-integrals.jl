include("hydrogenic_orbitals.jl")

using SymPy
using Polynomials

function Base.:(/)(p::Polynomials.Poly, q::Polynomials.Poly)
    all(q.a .>= 0) || sum(q.a) == 1 ||
        throw(ArgumentError("True polynomial division not implemented"))
    i = findfirst(isequal(1), q.a)
    all(p.a[1:(i-1)] .== 0) ||
        throw(ArgumentError("Polynomial division by $q would not result in a polynomial"))
    Polynomials.Poly(p.a[i:end], p.var)
end

function Ykcomp(n,ℓ,n′,ℓ′,k,Z=1//1)
    N,ρ,P=hydredwfn(Tuple,n,ℓ,Z)
    N′,ρ′,P′=hydredwfn(Tuple,n′,ℓ′,Z)

    a = (polyder(ρ)*polyder(ρ′)).a[1]
    r = Polynomials.Poly([0,1], P.var)
    f = P*P′
    g = k > 0 ? f*r^k : f/r^(abs(k))
    I = Polynomials.Poly([0], f.var)

    i = 1
    while true
        I += (isodd(i) ? 1 : -1)*g/a^i
        
        g = polyder(g)
        i += 1
        iszero(g) && break
    end
    I,r,a
end

function Y(n,ℓ,n′,ℓ′,k,Z=1//1)
    N,ρ,P=hydredwfn(Tuple,n,ℓ,Z)
    N′,ρ′,P′=hydredwfn(Tuple,n′,ℓ′,Z)
    
    r = symbols("r", real=true, positive=true)
    
    ee = exp(-(polyval(ρ, r)+polyval(ρ′, r))/2)
    PP = polyval(P*P′, r)

    Ysmaller = integrate(r^k*PP*ee, (r,0,r))/r^k
    Ygreater = integrate(PP*ee/(r^(k+1)), (r,r,oo))*r^(k+1)
    Yᵏ = N*N′*expand(Ysmaller + Ygreater)
end

function triangle_range(a,b)
    kmin = abs(a-b)
    if !iseven(kmin + a + b)
        kmin += 1
    end
    kmin:2:(a+b)
end

function generate_Yᵏ_references()
    nℓs = [(1,0),
           (2,0), (2,1),
           (3,0), (3,1), (3,2),
           (4,0), (4,1), (4,2), (4,3)]
    ks = 0:3

    open("exact_Yks.jl", "w") do file
        write(file, "exact_Yᵏs = Dict(")
        for ((n,ℓ), (n′,ℓ′), k) in Iterators.product(nℓs, nℓs, ks)
            k ∉ triangle_range(ℓ,ℓ′) && continue
            YY = Y(n,ℓ,n′,ℓ′,k)
            write(file, "(($n,$ℓ), ($(n′),$(ℓ′)), $k) => r -> $YY,\n")
        end
        write(file, ")")
    end
end

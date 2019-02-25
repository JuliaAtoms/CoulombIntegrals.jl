struct LazyCoulomb{T,B<:AbstractQuasiMatrix,I<:Integer} <: AbstractQuasiMatrix{T}
    R::B
    ℓ::I
end

LazyCoulomb(R::B, ℓ::I) where {T,B<:AbstractQuasiMatrix{T}, I<:Integer} =
    LazyCoulomb{T,B,I}(R,ℓ)

function Base.axes(LC::LazyCoulomb)
    a = axes(LC.R,1)
    a,a
end
Base.axes(LC::LazyCoulomb,i) = axes(LC.R,1)

function Base.size(LC::LazyCoulomb)
    m = size(LC.R,1)
    m,m
end
Base.size(LC::LazyCoulomb, i) = size(LC.R,1)

function Base.getindex(LC::LazyCoulomb{T,B,I},i::Integer,j::Integer) where {T,B,I}
    i ≠ j && return zero(T)
    R.x[i]
end

Base.replace_in_print_matrix(::LazyCoulomb, i::Integer, j::Integer, s::AbstractString) =
    i == j ? s : Base.replace_with_centered_mark(s)

const CoulombIntegral{T,B} = Mul{<:Tuple,<:Tuple{<:Adjoint{T,<:AbstractVector},<:QuasiAdjoint{T,B},<:LazyCoulomb{T,B},<:B,<:AbstractVector}}

K(ℓ::I,r::T,r̃::T) where {I<:Integer,T} = min(r,r̃)^ℓ/max(r,r̃)^(ℓ+1)

locs(B::FEDVR) = B.x
locs(B::AbstractFiniteDifferences) = FiniteDifferencesQuasi.locs(B)

function weight!(d, i, B::FEDVR)
    d[i,i] /= B.n[i]
end
function weight!(d, i, B::RadialDifferences)
    d[i,i] *= B.ρ
end
function weight!(d, i, B::NumerovFiniteDifferences)
    d[i,i] *= B.Δx
end

function Base.copyto!(dest::AbstractVector{T},
                      M::Mul{<:Tuple,<:Tuple{<:Adjoint{<:Any,<:AbstractVector},
                                             <:QuasiAdjoint{<:Any,<:Basis},
                                             <:Basis,
                                             <:AbstractMatrix,
                                             <:QuasiAdjoint{<:Any,<:Basis},
                                             <:Basis,
                                             <:AbstractVector}}) where {T,Basis<:AbstractQuasiMatrix{T}}
    axes(dest) == axes(M) || throw(DimensionMismatch("Incompatible axes"))
    u,A,B,O,C,D,v = M.factors
    A' == B == C' == D || throw(DimensionMismatch("Incompatible bases"))
    dest[1,1] = u*O*v
    dest
end

function Base.copyto!(dest::AbstractMatrix, M::CoulombIntegral{T,B}) where {T,B<:AbstractQuasiMatrix{T}}
    u,R₁,LC,R₂,v = M.factors
    @assert R₁' == R₂
    R = R₂
    r = locs(R)
    k = Diagonal(similar(r))
    k̂ = R⋆k⋆R'
    ukv = u⋆R₁⋆k̂⋆R₂⋆v
    for i in eachindex(r)
        k.diag .= K.(LC.ℓ,r,r[i])
        dest[i:i,i:i] .= ukv
        weight!(dest, i, R)
    end
    dest
end

function Base.similar(M::CoulombIntegral, ::Type{T}) where T
    B = M.factors[4]
    v = Vector{T}(undef, size(B,2))
    Diagonal(v)
end
LazyArrays.materialize(M::CoulombIntegral) = copyto!(similar(M, eltype(M)), M)

export LazyCoulomb

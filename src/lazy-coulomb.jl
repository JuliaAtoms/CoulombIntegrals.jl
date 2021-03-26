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

const CoulombIntegral{T} = Applied{<:Any, typeof(*),<:Tuple{
    Applied{<:Any, typeof(*), <:Tuple{
        <:Adjoint{T,<:AbstractVector},
        <:AbstractQuasiMatrix}},
    <:LazyCoulomb{T,<:AbstractQuasiMatrix},
    Applied{<:Any, typeof(*), <:Tuple{
        <:AbstractQuasiMatrix,
        <:AbstractVector}}}}

K(ℓ::I,r::T,r̃::T) where {I<:Integer,T} = min(r,r̃)^ℓ/max(r,r̃)^(ℓ+1)

function Base.copyto!(dest::AbstractVector{T}, M::CoulombIntegral{T}) where T
    u,LC,v = M.args
    R = LC.R
    r = locs(R)
    k = Diagonal(similar(r))

    Rv = R'v
    tmp = similar(v)

    tmp.args[2] .= 0

    kv = applied(*, k, Rv)

    for i in eachindex(r)
        k.diag .= K.(LC.ℓ,r,r[i])
        tmp.args[2] .= kv
        # u is already conjugated, hence we don't use the
        # transposition operator to calculate the dot product.
        dest[i:i] .= applied(*, u, tmp)
    end

    dest
end

function Base.similar(M::CoulombIntegral, ::Type{T}) where T
    LC = M.args[2]
    R = LC.R
    Vector{T}(undef, size(R,2))
end
LazyArrays.materialize(M::CoulombIntegral) = copyto!(similar(M, eltype(M)), M)

export LazyCoulomb

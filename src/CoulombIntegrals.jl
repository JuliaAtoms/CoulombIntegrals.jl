module CoulombIntegrals

using LinearAlgebra
using MatrixFactorizations

using LazyArrays
import LazyArrays: ⋆, materialize
import ContinuumArrays: axes
import ContinuumArrays.QuasiArrays: AbstractQuasiArray, AbstractQuasiMatrix, MulQuasiArray, QuasiAdjoint
using IntervalSets

using FEDVRQuasi
using FiniteDifferencesQuasi

using SpecialFunctions

const RadialOrbital{T,B<:AbstractQuasiMatrix} = Mul{<:Any, <:Tuple{B,<:AbstractVector{T}}}

locs(B::FEDVRQuasi.FEDVROrRestricted) = FEDVRQuasi.locs(B)
locs(B::AbstractFiniteDifferences) = FiniteDifferencesQuasi.locs(B)

include("analytical.jl")
include("lazy-coulomb.jl")
include("poisson.jl")

end # module

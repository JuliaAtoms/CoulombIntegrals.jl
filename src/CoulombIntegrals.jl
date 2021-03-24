module CoulombIntegrals

using LinearAlgebra

using LazyArrays
import LazyArrays: materialize
import ContinuumArrays: axes
import ContinuumArrays.QuasiArrays: AbstractQuasiArray, AbstractQuasiMatrix, MulQuasiArray, QuasiAdjoint
using IntervalSets

using CompactBases
import CompactBases: locs
using BlockBandedMatrices

using SpecialFunctions

const RadialOrbital{T,B<:AbstractQuasiMatrix} = Applied{<:Any, typeof(*), <:Tuple{B,<:AbstractVector{T}}}

include("analytical.jl")
include("lazy-coulomb.jl")
include("poisson.jl")
include("coulomb_repulsion_potential.jl")

end # module

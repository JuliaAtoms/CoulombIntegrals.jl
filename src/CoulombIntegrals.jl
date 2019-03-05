module CoulombIntegrals

using LinearAlgebra
using MatrixFactorizations

using LazyArrays
import LazyArrays: ⋆, materialize
import ContinuumArrays: axes
import ContinuumArrays.QuasiArrays: AbstractQuasiMatrix, MulQuasiArray, QuasiAdjoint

using FEDVRQuasi
using FiniteDifferencesQuasi

using SpecialFunctions

const RadialOrbital{T,B} = MulQuasiArray{T,1,<:Mul{<:Tuple, <:Tuple{B,<:AbstractVector}}}

include("analytical.jl")
include("lazy-coulomb.jl")
include("poisson.jl")

end # module

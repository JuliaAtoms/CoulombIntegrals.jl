module CoulombIntegrals

using LinearAlgebra
using IterativeSolvers
using AlgebraicMultigrid
using SparseArrays

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
include("laplacian_factorization.jl")
include("poisson.jl")

end # module

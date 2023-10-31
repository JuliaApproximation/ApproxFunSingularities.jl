module ApproxFunSingularitiesStaticArraysExt

using ApproxFunSingularities
# Specifying the full namespace is necessary because of https://github.com/JuliaLang/julia/issues/48533
# See https://github.com/JuliaStats/LogExpFunctions.jl/pull/63
using ApproxFunSingularities.ApproxFunBase
import ApproxFunSingularities.ApproxFunBase: coefficients
using ApproxFunSingularities.DomainSets
using StaticArrays

function coefficients(f::AbstractVector,
        sp::JacobiWeight{<:Any,<:Segment{<:SVector{2}}},
        S2::TensorSpace{<:Any,<:Any,<:EuclideanDomain{2}})

    coefficients(f,sp,JacobiWeight(0,0,S2))
end

end

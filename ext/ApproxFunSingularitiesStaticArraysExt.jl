module ApproxFunSingularitiesStaticArraysExt

using ApproxFunSingularities
using ApproxFunBase
import ApproxFunBase: coefficients
using DomainSets
using StaticArrays

function coefficients(f::AbstractVector,
        sp::JacobiWeight{<:Any,<:Segment{<:SVector{2}}},
        S2::TensorSpace{<:Any,<:Any,<:EuclideanDomain{2}})

    coefficients(f,sp,JacobiWeight(0,0,S2))
end

end

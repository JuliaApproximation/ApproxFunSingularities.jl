module ApproxFunSingularities

using DomainSets
using LinearAlgebra
using Reexport
using SpecialFunctions

@reexport using ApproxFunBase
@reexport using ApproxFunOrthogonalPolynomials


import ApproxFunBase: Fun, SumSpace, SubSpace, WeightSpace, NoSpace,
            IntervalOrSegment, ConcreteMultiplication, ConcreteDerivative,
            MultiplicationWrapper, ConversionWrapper, DerivativeWrapper,
            Evaluation, Conversion, defaultConversion, defaultcoefficients,
            Multiplication, Derivative, bandwidths,
            ConcreteEvaluation, ConcreteDefiniteLineIntegral,
            ConcreteDefiniteIntegral, ConcreteIntegral,
            DefiniteLineIntegral, DefiniteIntegral, ConcreteDefiniteIntegral,
            ConcreteDefiniteLineIntegral, maxspace, hasconversion,
            Space, spacescompatible, canonicalspace, domain, setdomain,
            domainscompatible, domainspace, rangespace,
            union_rule, conversion_rule, maxspace_rule, conversion_type,
            linesum, differentiate, integrate, linebilinearform, bilinearform,
            subspace_coefficients, sumspacecoefficients,
            specialfunctionnormalizationpoint, Segment, IntervalOrSegmentDomain,
            eps, arclength, complexlength, tocanonical, fromcanonicalD,
            tocanonicalD, canonicaldomain, setcanonicaldomain,
            coefficients, isconvertible, SpaceOperator, cfstype, mobius, roots,
            splitatroots, domaintype, rangetype, weight, isapproxinteger,
            dotu, components, promoterangespace, ∞,
            assert_integer, SpecialEvalPtType, isleftendpoint, isrightendpoint, evaluation_point,
            @calculus_operator, ConcreteConversion, InterlaceOperator_Diagonal, UnsetSpace,
            choosedomainspace, mean, bandwidthssum

import ApproxFunOrthogonalPolynomials: order

import IntervalSets: rightendpoint, leftendpoint, Domain

import DomainSets: ChebyshevInterval, boundary, dimension

using BlockBandedMatrices: blockbandwidths, subblockbandwidths

import Base: convert, getindex, *, /, ^,
            show, sum, cumsum, complex, sqrt, abs, in, first, last,
            union, isapprox, zeros, one, length, ones, exp, log

using HalfIntegers
using OddEvenIntegers

const HalfOddInteger{T<:Integer} = Half{Odd{T}}

strictconvert(::Type{T}, x) where {T} = convert(T, x)::T

isapproxhalfoddinteger(a) = isapproxinteger(a + half(Odd(1)))
isapproxhalfoddinteger(::HalfOddInteger) = true
isapproxhalfoddinteger(::Integer) = false

isequalminhalf(x) = x == -0.5
isequalminhalf(::Integer) = false

const MaybeNormalizedPolynomialSpace{S<:PolynomialSpace} = Union{S, NormalizedPolynomialSpace{S}}

include("divide_singularity.jl")
include("JacobiWeight.jl")
include("JacobiWeightOperators.jl")
include("JacobiWeightChebyshev.jl")
include("LogWeight.jl")
include("ExpWeight.jl")
include("fractionalcalculus.jl")

/(c::Number,f::Fun{<:Ultraspherical}) = c/Fun(f,Chebyshev(domain(f)))
/(c::Number,f::Fun{<:PolynomialSpace{<:IntervalOrSegment}}) = c/Fun(f,Chebyshev(domain(f)))
/(c::Number,f::Fun{<:Chebyshev}) = setdomain(c/setcanonicaldomain(f),domain(f))

scaleshiftdomain(f::Fun,sc,sh) = setdomain(f,sc .* domain(f) .+ sh)

function /(c::Number,f::Fun{<:Chebyshev{<:IntervalOrSegment}})
    fc = setcanonicaldomain(f)
    d=domain(f)
    # if domain f is small then the pts get projected in
    tol = 200eps(promote_type(typeof(c),cfstype(f)))*norm(f.coefficients,1)

    # we prune out roots at the boundary first
    if ncoefficients(f) == 1
        return Fun(c/f.coefficients[1],space(f))
    elseif ncoefficients(f) == 2
        if isempty(roots(f))
            return \(Multiplication(f,space(f)),c;tolerance=0.05tol)
        elseif isapprox(fc.coefficients[1],fc.coefficients[2])
            # we check directly for f*(1+x)
            return Fun(JacobiWeight(-1,0,space(f)),[c/fc.coefficients[1]])
        elseif isapprox(fc.coefficients[1],-fc.coefficients[2])
            # we check directly for f*(1-x)
            return Fun(JacobiWeight(0,-1,space(f)),[c/fc.coefficients[1]])
        else
            # we need to split at the only root
            return c/splitatroots(f)
        end
    elseif abs(first(fc)) ≤ tol
        #left root
        g=divide_singularity((1,0),fc)
        p=c/g
        x = Fun(identity,domain(p))
        return scaleshiftdomain(p/(1+x),complexlength(d)/2,mean(d) )
    elseif abs(last(fc)) ≤ tol
        #right root
        g=divide_singularity((0,1),fc)
        p=c/g
        x=Fun(identity,domain(p))
        return scaleshiftdomain(p/(1-x),complexlength(d)/2,mean(d) )
    else
        r = roots(fc)

        if length(r) == 0
            return \(Multiplication(f,space(f)),c;tolerance=0.05tol)
        elseif abs(last(r)+1.0)≤tol  # double check
            #left root
            g=divide_singularity((1,0),fc)
            p=c/g
            x=Fun(identity,domain(p))
            return scaleshiftdomain(p/(1+x),complexlength(d)/2,mean(d) )
        elseif abs(last(r)-1.0)≤tol  # double check
            #right root
            g=divide_singularity((0,1),fc)
            p=c/g
            x=Fun(identity,domain(p))
            return scaleshiftdomain(p/(1-x),complexlength(d)/2,mean(d) )
        else
            # no roots on the boundary
            return c/splitatroots(f)
        end
    end
end

function ^(f::Fun{<:PolynomialSpace}, k::Real)
    T = cfstype(f)
    RT = real(T)
    # Need to think what to do if this is ever not the case..
    sp = space(f)
    fc = setcanonicaldomain(f) #Project to interval
    csp = space(fc)

    r = sort(roots(fc))
    #TODO divideatroots
    @assert length(r) <= 2

    if length(r) == 0
        setdomain(Fun((x->x^k) ∘ fc,csp),domain(f))  # using ∘ supports fast transforms for fc
    elseif length(r) == 1
        @assert isapprox(abs(r[1]),1)

        if isapprox(r[1], 1)
            Fun(JacobiWeight(zero(RT),k,sp),coefficients(divide_singularity(true,fc)^k,csp))
        else
            Fun(JacobiWeight(k,zero(RT),sp),coefficients(divide_singularity(false,fc)^k,csp))
        end
    else
        @assert isapprox(r[1],-1)
        @assert isapprox(r[2],1)

        Fun(JacobiWeight(k,k,sp),coefficients(divide_singularity(fc)^k,csp))
    end
end


# function log{MS<:MappedSpace}(f::Fun{MS})
#     g=log(Fun(f.coefficients,space(f).space))
#     Fun(g.coefficients,MappedSpace(domain(f),space(g)))
# end

# project first to [-1,1] to avoid issues with
# complex derivative
function log(f::Fun{<:PolynomialSpace{<:ChebyshevInterval}})
    r = sort(roots(f))
    #TODO divideatroots
    @assert length(r) <= 2

    if length(r) == 0
        cumsum(differentiate(f)/f)+log(first(f))
    elseif length(r) == 1
        @assert isapprox(abs(r[1]),1)

        if isapprox(r[1],1.)
            g=divide_singularity(true,f)
            lg=Fun(LogWeight(0.,1.,Chebyshev()),[1.])
            if isapprox(g,1.)  # this means log(g)~0
                lg
            else # log((1-x)) + log(g)
                lg⊕log(g)
            end
        else
            g=divide_singularity(false,f)
            lg=Fun(LogWeight(1.,0.,Chebyshev()),[1.])
            if isapprox(g,1.)  # this means log(g)~0
                lg
            else # log((1+x)) + log(g)
                lg⊕log(g)
            end
       end
    else
        @assert isapprox(r[1],-1)
        @assert isapprox(r[2],1)

        g=divide_singularity(f)
        lg=Fun(LogWeight(1.,1.,Chebyshev()),[1.])
        if isapprox(g,1.)  # this means log(g)~0
            lg
        else # log((1+x)) + log(g)
            lg⊕log(g)
        end
    end
end

# JacobiWeight explodes, we want to ensure the solution incorporates the fact
# that exp decays rapidly
exp(f::Fun{<:JacobiWeight}) = setdomain(exp(setdomain(f, ChebyshevInterval())), domain(f))
function exp(f::Fun{<:JacobiWeight{<:Any,<:ChebyshevInterval}})
    S=space(f)
    q=Fun(S.space,f.coefficients)
    if isapprox(S.α,0.) && isapprox(S.β,0.)
        exp(q)
    elseif S.β < 0 && isapprox(first(q),0.)
        # this case can remove the exponential decay
        exp(Fun(f,JacobiWeight(S.β+1,S.α,S.space)))
    elseif S.α < 0 && isapprox(last(q),0.)
        exp(Fun(f,JacobiWeight(S.β,S.α+1,S.space)))
    elseif S.β > 0 && isapproxinteger(S.β)
        exp(Fun(f,JacobiWeight(0.,S.α,S.space)))
    elseif S.α > 0 && isapproxinteger(S.α)
        exp(Fun(f,JacobiWeight(S.β,0.,S.space)))
    else
        #find normalization point
        xmax,opfxmax,opmax=specialfunctionnormalizationpoint(exp,real,f)

        if S.α < 0 && S.β < 0
            # provided both are negative, we get exponential decay on both ends
            @assert real(first(q)) < 0 && real(last(q)) < 0
            s=JacobiWeight(2.,2.,domain(f))
        elseif S.β < 0 && isapprox(S.α,0.)
            @assert real(first(q)) < 0
            s=JacobiWeight(2.,0.,domain(f))
        elseif S.α < 0 && isapprox(S.β,0.)
            @assert real(last(q)) < 0
            s=JacobiWeight(0.,2.,domain(f))
        else
            error("exponential of fractional power, not implemented")
        end

        D=Derivative(s)
        B=Evaluation(s,xmax)

        \([B,D-f'], [opfxmax,0.]; tolerance=eps(cfstype(f))*opmax)
    end
end


## Root finding for JacobiWeight expansion

# Add endpoints for JacobiWeight
# TODO: what about cancellation?
function roots(f::Fun{<:JacobiWeight})
    sp=space(f)
    d=domain(sp)
    rts=roots(Fun(sp.space,f.coefficients))
    if sp.β > 0
        rts=[leftendpoint(d);rts]
    end
    if sp.α > 0
        rts=[rts;rightendpoint(d)]
    end
    rts
end

for Func in (:DefiniteIntegral,:DefiniteLineIntegral)
    @eval begin
        #TODO: this may be misleading
        $Func(d::IntervalOrSegment) = $Func(JacobiWeight(-0.5,-0.5,Chebyshev(d)))
        function $Func(α::Number,β::Number,d::IntervalOrSegment)
            @assert α == β
            @assert round(Int,α+.5) == α+.5
            @assert round(Int,α+.5) >= 0
            $Func(JacobiWeight(α,β,Ultraspherical(round(Int,α+.5),d)))
        end
        $Func(α::Number,β::Number) = $Func(α,β,ChebyshevInterval())
    end
end

## Integration
function integrate(f::Fun{<:LaguerreWeight{<:Laguerre}})
    α = space(f).α;
    n = length(f.coefficients);
    if space(f).space.α != α
        throw(ArgumentError("`integrate` is applicable if only LaguerreWeight parameter and Laguerre parameter are equal."))
    else
        if n == 0
            Fun(0)
        else
            if !isinteger(α) || α == 2
                if n == 1
                    f̃ = Fun(f, JacobiWeight(α, 0, Chebyshev(Ray())));
                    g = integrate(f̃);
                    g = g - last(g)
                else
                    if f.coefficients[1] == 0
                        Fun(WeightedLaguerre(α + 1), f.coefficients[2:end] ./ (1:n-1))
                    else
                        f₀ = Fun(WeightedLaguerre(α), [f.coefficients[1]]);
                        f₀ = Fun(f₀, JacobiWeight(α, 0, Chebyshev(Ray())));
                        g₀ = integrate(f₀);
                        g₀ = g₀ - last(g₀);
                        g = Fun(WeightedLaguerre(α + 1), f.coefficients[2:end] ./ (1:n-1));
                        g₀ + g
                    end
                end
            else
                if n == 1
                    f̃ = Fun(f, Chebyshev(Ray()));
                    g = integrate(f̃);
                    g = g - last(g)
                else
                    if f.coefficients[1] == 0
                        Fun(WeightedLaguerre(α + 1), f.coefficients[2:end] ./ (1:n-1))
                    else
                        f₀ = Fun(WeightedLaguerre(α), [f.coefficients[1]]);
                        f₀ = Fun(f₀, Chebyshev(Ray()));
                        g₀ = integrate(f₀);
                        g₀ = g₀ - last(g₀);
                        g = Fun(WeightedLaguerre(α + 1), f.coefficients[2:end] ./ (1:n-1));
                        g₀ + g
                    end
                end
            end
        end
    end
end

if !isdefined(Base, :get_extension)
    include("../ext/ApproxFunSingularitiesStaticArraysExt.jl")
end

end

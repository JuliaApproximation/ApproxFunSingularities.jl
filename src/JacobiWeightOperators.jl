## Calculus

Base.sum(f::Fun{<:JacobiWeight{<:Chebyshev}}) =
    sum(setdomain(f,canonicaldomain(f))*fromcanonicalD(f))

linesum(f::Fun{<:JacobiWeight{<:Chebyshev}}) =
    linesum(setdomain(f,canonicaldomain(f))*abs(fromcanonicalD(f)))

for (Func,Len) in ((:(Base.sum),:complexlength), (:linesum,:arclength))
    @eval begin
        function $Func(f::Fun{<:JacobiWeight{<:Chebyshev,<:IntervalOrSegment}})
            tol=1e-10
            d,β,α,n=domain(f),f.space.β,f.space.α,ncoefficients(f)
            n == 0 && return zero(cfstype(f))*$Len(d)
            g=Fun(space(f).space,f.coefficients)
            if β ≤ -1.0 && abs(first(g))≤tol
                $Func(increase_jacobi_parameter(-1,f))
            elseif α ≤ -1.0 && abs(last(g))≤tol
                $Func(increase_jacobi_parameter(+1,f))
            elseif β ≤ -1.0 || α ≤ -1.0
                fs = Fun(f.space.space,f.coefficients)
                return Inf*0.5*$Len(d)*(sign(fs(leftendpoint(d)))+sign(fs(rightendpoint(d))))/2
            elseif β == α == -0.5
                return 0.5*$Len(d)*f.coefficients[1]*π
            elseif β == α == 0.5
                return 0.5*$Len(d)*(n ≤ 2 ? f.coefficients[1]/2 : f.coefficients[1]/2 - f.coefficients[3]/4)*π
            elseif β == 0.5 && α == -0.5
                return 0.5*$Len(d)*(n == 1 ? f.coefficients[1] : f.coefficients[1] + f.coefficients[2]/2)*π
            elseif β == -0.5 && α == 0.5
                return 0.5*$Len(d)*(n == 1 ? f.coefficients[1] : f.coefficients[1] - f.coefficients[2]/2)*π
            else
                c = zeros(cfstype(f),n)
                c[1] = 2^(β+α+1)*gamma(β+1)*gamma(α+1)/gamma(β+α+2)
                if n > 1
                    c[2] = c[1]*(β-α)/(β+α+2)
                    for i=1:n-2
                        c[i+2] = (2(β-α)*c[i+1]-(β+α-i+2)*c[i])/(β+α+i+2)
                    end
                end
                return 0.5*$Len(d)*dotu(f.coefficients,c)
            end
        end
        $Func(f::Fun{<:JacobiWeight{<:MaybeNormalizedPolynomialSpace}}) =
            $Func(Fun(f,JacobiWeight(space(f).β, space(f).α, Chebyshev(domain(f)))))
    end
end

function differentiate(f::Fun{<:JacobiWeight{<:Any,<:IntervalOrSegment}})
    S=f.space
    d=domain(f)
    ff=Fun(S.space,f.coefficients)
    zeroT = zero(S.α-1)
    if S.β==S.α==0
        u=differentiate(ff)
        Fun(JacobiWeight(zeroT,zeroT,space(u)),u.coefficients)
    elseif S.β==0
        x=Fun(identity,d)
        M=tocanonical(d,x)
        Mp=tocanonicalD(d,leftendpoint(d))
        u=-Mp*S.α*ff +(1-M).*differentiate(ff)
        Fun(JacobiWeight(zeroT,S.α-1,space(u)),u.coefficients)
    elseif S.α==0
        x=Fun(identity,d)
        M=tocanonical(d,x)
        Mp=tocanonicalD(d,leftendpoint(d))
        u=Mp*S.β*ff +(1+M).*differentiate(ff)
        Fun(JacobiWeight(S.β-1,zeroT,space(u)),u.coefficients)
    else
        x=Fun(identity,d)
        M=tocanonical(d,x)
        Mp=tocanonicalD(d,leftendpoint(d))
        u=(Mp*S.β)*(1-M).*ff- (Mp*S.α)*(1+M).*ff +(1-M.^2).*differentiate(ff)
        Fun(JacobiWeight(S.β-1,S.α-1,space(u)),u.coefficients)
    end
end

function integrate(f::Fun{<:JacobiWeight{<:Any,<:IntervalOrSegment,<:Any,TT}}) where {TT<:Real}
    S=space(f)
    # we integrate by solving u'=f
    tol=1e-10
    g=Fun(S.space,f.coefficients)
    if S.β ≈ 0 && S.α ≈ 0
        integrate(g)
    elseif S.β ≤ -1 && abs(first(g)) ≤ tol
        integrate(increase_jacobi_parameter(-1,f))
    elseif S.α ≤ -1 && abs(last(g)) ≤ tol
        integrate(increase_jacobi_parameter(+1,f))
    elseif S.β ≈ -1 && S.α ≈ -1
        error("Implement")
    elseif S.β ≈ -1 && S.α ≈ 0
        p=first(g)  # first value without weight
        fp = Fun(f-Fun(S,[p]),S.space)  # Subtract out right value and divide singularity via conversion
        d=domain(f)
        Mp=tocanonicalD(d,leftendpoint(d))
        integrate(fp) ⊕ Fun(LogWeight(1.,0.,S.space),[p/Mp])
    elseif S.β ≈ -1 && S.α > 0 && isapproxinteger(S.α)
        # convert to zero case and integrate
        integrate(Fun(f,JacobiWeight(S.β,0,S.space)))
    elseif S.α ≈ -1 && S.β ≈ 0
        p=last(g)  # last value without weight
        fp = Fun(f-Fun(S,[p]),S.space)  # Subtract out right value and divide singularity via conversion
        d=domain(f)
        Mp=tocanonicalD(d,leftendpoint(d))
        integrate(fp) ⊕ Fun(LogWeight(zero(TT),one(TT),S.space),[-p/Mp])
    elseif isapprox(S.α,-1) && S.β > 0 && isapproxinteger(S.β)
        # convert to zero case and integrate
        integrate(Fun(f,JacobiWeight(zero(TT),S.α,S.space)))
    elseif S.β ≈ 0
        D = Derivative(JacobiWeight(S.β, S.α+1, S.space))
        D\f   # this happens to pick out a smooth solution
    elseif S.α ≈ 0
        D = Derivative(JacobiWeight(S.β+1, S.α, S.space))
        D\f   # this happens to pick out a smooth solution
    elseif isapproxinteger(S.β) || isapproxinteger(S.α)
        D = Derivative(JacobiWeight(S.β+1, S.α+1, S.space))
        D\f   # this happens to pick out a smooth solution
    else
        s=sum(f)
        if abs(s)<1E-14
            D=Derivative(JacobiWeight(S.β+1, S.α+1, S.space))
            \(D,f; tolerance=1E-14)  # if the sum is 0 we don't get step-like behaviour
        else
            # we normalized so it sums to zero, and so backslash works
            w = Fun(x->exp(-40x^2),81)
            w1 = Fun(S,coefficients(w))
            w2 = Fun(x->w1(x),domain(w1))
            c  = s/sum(w1)
            v  = f-w1*c
            (c*integrate(w2)) ⊕ integrate(v)
        end
    end
end

function Base.cumsum(f::Fun{<:JacobiWeight{<:Any,<:IntervalOrSegmentDomain}})
    g=integrate(f)
    S=space(f)

    if (S.β==0 && S.α==0) || S.β>-1
        g-first(g)
    else
        @warn "Function is not integrable at left endpoint. Returning a non-normalized indefinite integral."
        g
    end
end


## Operators


function jacobiweightDerivative(S::JacobiWeight{<:Any,<:IntervalOrSegment})
    d = domain(S)
    # map to canonical
    Mp=fromcanonicalD(d,leftendpoint(d))
    DD=jacobiweightDerivative(setdomain(S,ChebyshevInterval()))

    return DerivativeWrapper(SpaceOperator(DD.op.op,S,setdomain(rangespace(DD),d))/Mp,1)
end

function jacobiweightDerivative(S::JacobiWeight{<:Any,<:ChebyshevInterval})
    d=domain(S)

    zeroT = zero(S.α-1)
    oneT = oneunit(zeroT)
    DSsp = Derivative(S.space)

    if S.β == S.α == 0
        DerivativeWrapper(SpaceOperator(DSsp,S,
                JacobiWeight(zeroT,zeroT,rangespace(DSsp))),1)
    elseif S.β == 0
        w = Fun(JacobiWeight(zeroT,oneT,ConstantSpace(d)),[1.0])

        DDβ0 = -S.α + w*DSsp
        rsβ0 = JacobiWeight(zeroT,S.α-1,rangespace(DDβ0))
        DerivativeWrapper(SpaceOperator(DDβ0,S,rsβ0),1)
    elseif S.α == 0
        w = Fun(JacobiWeight(oneT,zeroT,ConstantSpace(d)),[1.0])

        DDα0 = S.β + w*DSsp
        rsα0 = JacobiWeight(S.β-1,zeroT,rangespace(DDα0))
        DerivativeWrapper(SpaceOperator(DDα0,S,rsα0),1)
    else
        w = Fun(JacobiWeight(oneT,oneT,ConstantSpace(d)),[1.0])
        x=Fun()

        DD = S.β*(1-x) - S.α*(1+x) + w*DSsp
        rs = JacobiWeight(S.β-1,S.α-1,rangespace(DD))
        DerivativeWrapper(SpaceOperator(DD,S,rs),1)
    end
end

Derivative(S::JacobiWeight{<:Any,<:IntervalOrSegment}) = jacobiweightDerivative(S)

function Derivative(S::JacobiWeight{<:Any,<:IntervalOrSegment}, k::Number)
    assert_integer(k)
    if k==1
        Derivative(S)
    else
        D=Derivative(S)
        Dk = TimesOperator(Derivative(rangespace(D),k-1), D)
        DerivativeWrapper(Dk, k, S)
    end
end




## Multiplication

#Left multiplication. Here, S is considered the domainspace and we determine rangespace accordingly.

function Multiplication(f::Fun{<:JacobiWeight},S::JacobiWeight)
    M=Multiplication(Fun(space(f).space,f.coefficients),S.space)
    if space(f).β+S.β==space(f).α+S.α==0
        rsp=rangespace(M)
    else
        rsp=JacobiWeight(space(f).β+S.β,space(f).α+S.α,rangespace(M))
    end
    MultiplicationWrapper(f,SpaceOperator(M,S,rsp))
end

function Multiplication(f::Fun, S::JacobiWeight)
    M=Multiplication(f,S.space)
    rsp=JacobiWeight(S.β,S.α,rangespace(M))
    MultiplicationWrapper(f,SpaceOperator(M,S,rsp))
end

function Multiplication(f::Fun{<:JacobiWeight},S::MaybeNormalizedPolynomialSpace)
    M=Multiplication(Fun(space(f).space,f.coefficients),S)
    rsp=JacobiWeight(space(f).β,space(f).α,rangespace(M))
    MultiplicationWrapper(f,SpaceOperator(M,S,rsp))
end

#Right multiplication. Here, S is considered the rangespace and we determine domainspace accordingly.

function Multiplication(S::JacobiWeight, f::Fun{<:JacobiWeight})
    M=Multiplication(Fun(space(f).space,f.coefficients),S.space)
    dsp=canonicalspace(JacobiWeight(S.β-space(f).β,S.α-space(f).α,rangespace(M)))
    MultiplicationWrapper(f,SpaceOperator(M,dsp,S))
end

function Multiplication(S::JacobiWeight, f::Fun)
    M=Multiplication(f,S.space)
    dsp=JacobiWeight(S.β,S.α,rangespace(M))
    MultiplicationWrapper(f,SpaceOperator(M,dsp,S))
end

# function Multiplication{D<:JacobiWeight,T,V,ID<:IntervalOrSegment}(S::Space{V,D},f::Fun{ID,T})
#     M=Multiplication(Fun(f.coefficients,space(f).space),S)
#     dsp=JacobiWeight(-space(f).β,-space(f).α,rangespace(M))
#     MultiplicationWrapper(f,SpaceOperator(M,dsp,S))
# end

function _maxspace_rule(A::JacobiWeight, B::JacobiWeight, f)
    if domainscompatible(A,B) && isapproxinteger(A.β-B.β) && isapproxinteger(A.α-B.α)
        ms = f(A.space,B.space)
        if min(A.β,B.β) == 0 && min(A.α,B.α) == 0
            return ms
        else
            return JacobiWeight(min(A.β,B.β),min(A.α,B.α),ms)
        end
    end
    NoSpace()
end

## Conversion
for (OPrule,OP) in ((:maxspace_rule,:maxspace),(:union_rule,:union))
    @eval begin
        $OPrule(A::JacobiWeight, B::JacobiWeight) = _maxspace_rule(A, B, $OP)
        $OPrule(A::JacobiWeight, B::Space{<:IntervalOrSegmentDomain}) = $OPrule(A,JacobiWeight(0,0,B))
    end
end


for FUNC in (:hasconversion,:isconvertible)
    @eval begin
        $FUNC(A::JacobiWeight{<:Any,D}, B::JacobiWeight{<:Any,D}) where {D<:IntervalOrSegmentDomain} =
            isapproxinteger(A.β-B.β) &&
            isapproxinteger(A.α-B.α) && A.β ≥ B.β && A.α ≥ B.α && $FUNC(A.space,B.space)

        $FUNC(A::JacobiWeight{<:Any,D},B::Space{D}) where {D<:IntervalOrSegmentDomain} =
            $FUNC(A,JacobiWeight(0,0,B))
        $FUNC(B::Space{D},A::JacobiWeight{<:Any,D}) where {D<:IntervalOrSegmentDomain} =
            $FUNC(JacobiWeight(0,0,B),A)
        $FUNC(A::SumSpace{<:Any,D},B::JacobiWeight{<:Any,D}) where {D<:IntervalOrSegmentDomain} =
            all(s -> $FUNC(s, B), A.spaces)
    end
end


# return the space that has banded Conversion to the other, or NoSpace

function conversion_rule(A::JacobiWeight, B::JacobiWeight)
    if isapproxinteger(A.β-B.β) && isapproxinteger(A.α-B.α)
        ct=conversion_type(A.space,B.space)
        ct==NoSpace() ? NoSpace() : JacobiWeight(max(A.β,B.β),max(A.α,B.α),ct)
    else
        NoSpace()
    end
end

conversion_rule(A::JacobiWeight, B::Space{<:IntervalOrSegmentDomain}) =
    conversion_type(A, JacobiWeight(0,0,B))


# override defaultConversion instead of Conversion to avoid ambiguity errors
function defaultConversion(A::JacobiWeight{<:Any,<:IntervalOrSegmentDomain},
        B::JacobiWeight{<:Any,<:IntervalOrSegmentDomain})

    @assert isapproxinteger(A.β-B.β) && isapproxinteger(A.α-B.α)

    if isapprox(A.β,B.β) && isapprox(A.α,B.α)
        ConversionWrapper(SpaceOperator(Conversion(A.space,B.space),A,B))
    else
        @assert A.β≥B.β && A.α≥B.α
        # first check if a multiplication by JacobiWeight times B.space is overloaded
        # this is currently designed for Jacobi multiplied by (1-x), etc.
        βdif,αdif=round(Int,A.β-B.β),round(Int,A.α-B.α)
        d=domain(A)
        M=Multiplication(jacobiweight(βdif,αdif,d), A.space)

        if rangespace(M)==JacobiWeight(βdif,αdif,A.space)
            # M is the default, so we should use multiplication by polynomials instead
            x=Fun(identity,d)
            y=mobius(d,x)   # we use mobius instead of tocanonical so that it works for Funs
            m=(1+y)^βdif*(1-y)^αdif
            MC=promoterangespace(Multiplication(m,A.space), B.space)

            ConversionWrapper(SpaceOperator(MC,A,B))# Wrap the operator with the correct spaces
        else
            ConversionWrapper(SpaceOperator(promoterangespace(M,B.space), A,B))
        end
    end
end

defaultConversion(A::Space{<:IntervalOrSegmentDomain,<:Real},
        B::JacobiWeight{<:Any,<:IntervalOrSegmentDomain}) =
    ConversionWrapper(SpaceOperator(Conversion(JacobiWeight(0,0,A),B),A,B))

defaultConversion(A::JacobiWeight{<:Any,<:IntervalOrSegmentDomain},
        B::Space{<:IntervalOrSegmentDomain,<:Real}) =
    ConversionWrapper(SpaceOperator(Conversion(A,JacobiWeight(0,0,B)),A,B))






## Evaluation

function getindex(op::ConcreteEvaluation{<:JacobiWeight,<:SpecialEvalPtType}, kr::AbstractRange)
    x = evaluation_point(op)
    if isleftendpoint(x)
        _getindex_eval_leftendpoint(op, kr)
    elseif isrightendpoint(x)
        _getindex_eval_rightendpoint(op, kr)
    else
        throw(ArgumentError("Evaluation is supported only at the leftendpoint/rightendpoint of the domain"))
    end
end

function _getindex_eval_leftendpoint(op::ConcreteEvaluation, kr::AbstractRange)
    S=op.space
    @assert op.order ≤ 1
    d=domain(op)

    @assert S.β ≥ 0
    if S.β==0
        if op.order==0
            2^S.α*getindex(Evaluation(S.space,op.x),kr)
        else #op.order ===1
            @assert isa(d,IntervalOrSegment)
            2^S.α * Evaluation(S.space,op.x,1)[kr] -
                (tocanonicalD(d,leftendpoint(d))*S.α*2^(S.α-1)) * Evaluation(S.space,op.x)[kr]
        end
    else
        @assert op.order==0
        zeros(eltype(op), length(kr))
    end
end

function _getindex_eval_rightendpoint(op::ConcreteEvaluation, kr::AbstractRange)
    S=op.space
    @assert op.order<=1
    d=domain(op)

    @assert S.α>=0
    if S.α==0
        if op.order==0
            2^S.β*getindex(Evaluation(S.space,op.x),kr)
        else #op.order ===1
            @assert isa(d,IntervalOrSegment)
            2^S.β * Evaluation(S.space,op.x,1)[kr] +
                tocanonicalD(d,leftendpoint(d))*S.β*2^(S.β-1) * Evaluation(S.space,op.x)[kr]
        end
    else
        @assert op.order==0
        zeros(eltype(op), length(kr))
    end
end


## Definite Integral

for (Func,Len,Sum) in ((:DefiniteIntegral,:complexlength,:sum),(:DefiniteLineIntegral,:arclength,:linesum))
    ConcFunc = Symbol(:Concrete, Func)

    @eval begin
        $Func(S::JacobiWeight{<:Any,<:IntervalOrSegment}) = $ConcFunc(S)

        getindex(Σ::$ConcFunc, k::Integer) = eltype(Σ)($Sum(Fun(domainspace(Σ),[zeros(eltype(Σ),k-1);1])))

        function getindex(Σ::$ConcFunc{<:JacobiWeight{<:Ultraspherical{<:Any,D,R},D,R},T},
                k::Integer) where {D<:IntervalOrSegment,R,T<:Real}
            λ = order(domainspace(Σ).space)
            dsp = domainspace(Σ)
            d = domain(Σ)
            C = $Len(d)/2

            if isequalminhalf(dsp.β - λ) && isequalminhalf(dsp.α - λ)
                k == 1 ? strictconvert(T,C*gamma(λ+one(T)/2)*gamma(one(T)/2)/gamma(λ+one(T))) : zero(T)
            else
                strictconvert(T,$Sum(Fun(dsp,[zeros(T,k-1);1])))
            end
        end

        function getindex(Σ::$ConcFunc{<:JacobiWeight{<:Ultraspherical{<:Any,D,R},D,R},T},
                kr::AbstractRange) where {D<:IntervalOrSegment,R,T<:Real}
            λ = order(domainspace(Σ).space)
            dsp = domainspace(Σ)
            d = domain(Σ)
            C = $Len(d)/2

            if isequalminhalf(dsp.β - λ) && isequalminhalf(dsp.α - λ)
                T[k == 1 ? C*gamma(λ+one(T)/2)*gamma(one(T)/2)/gamma(λ+one(T)) : zero(T) for k=kr]
            else
                T[$Sum(Fun(dsp,[zeros(T,k-1);1])) for k=kr]
            end
        end

        function bandwidths(Σ::$ConcFunc{<:JacobiWeight{<:Ultraspherical{<:Any,D,R},D,R}}) where {D<:IntervalOrSegment,R}
            λ = order(domainspace(Σ).space)
            β,α = domainspace(Σ).β,domainspace(Σ).α
            if isapproxhalfoddinteger(β-λ) && β==α && λ ≤ ceil(Int,β)
                0,2*(ceil(Int,β)-λ)
            else
                0,∞
            end
        end

        function getindex(Σ::$ConcFunc{<:JacobiWeight{Chebyshev{D,R},D,R},T},
                k::Integer) where {D<:IntervalOrSegment,R,T<:Real}
            dsp = domainspace(Σ)
            d = domain(Σ)
            C = $Len(d)/2

            if isequalminhalf(dsp.β) && isequalminhalf(dsp.α)
                k == 1 ? strictconvert(T,C*π) : zero(T)
            else
                strictconvert(T,$Sum(Fun(dsp,[zeros(T,k-1);1])))
            end
        end

        function getindex(Σ::$ConcFunc{<:JacobiWeight{Chebyshev{D,R},D,R},T},
                kr::AbstractRange) where {D<:IntervalOrSegment,R,T<:Real}
            dsp = domainspace(Σ)
            d = domain(Σ)
            C = $Len(d)/2

            if isequalminhalf(dsp.β) && isequalminhalf(dsp.α)
                T[k == 1 ? C*π : zero(T) for k=kr]
            else
                T[$Sum(Fun(dsp,[zeros(T,k-1);1])) for k=kr]
            end
        end

        function bandwidths(Σ::$ConcFunc{<:JacobiWeight{Chebyshev{D,R},D,R}}) where {D<:IntervalOrSegment,R}
            β,α = domainspace(Σ).β,domainspace(Σ).α
            if isapproxhalfoddinteger(β) && β==α && 0 ≤ ceil(Int,β)
                0,2ceil(Int,β)
            else
                0,∞
            end
        end
    end
end

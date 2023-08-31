export JacobiWeight, WeightedJacobi



"""
    JacobiWeight(β,α,s::Space)

weights a space `s` by a Jacobi weight, which on `-1..1`
is `(1+x)^β*(1-x)^α`.
For other domains, the weight is inferred by mapping to `-1..1`.
"""
struct JacobiWeight{S,DD,RR,T<:Real} <: WeightSpace{S,DD,RR}
    β::T
    α::T
    space::S
    function JacobiWeight{S,DD,RR,T}(β::T,α::T,space::S) where {S<:Space,DD,RR,T}
        if space isa JacobiWeight
            new(β+space.β, α+space.α, space.space)
        else
            new(β, α, space)
        end
    end
end

const WeightedJacobi{D,R} = JacobiWeight{<:Jacobi{D,R},D,R,R}

JacobiWeight{S,DD,RR,T}(β,α,space::Space) where {S,DD,RR,T} =
    JacobiWeight{S,DD,RR,T}(strictconvert(T,β)::T, strictconvert(T,α)::T, convert(S,space)::S)

JacobiWeight(a::Number, b::Number, d::Space) =
    JacobiWeight{typeof(d),domaintype(d),rangetype(d),promote_type(eltype(a),eltype(b))}(a,b,d)
JacobiWeight(β::Number, α::Number, d::JacobiWeight) =  JacobiWeight(β+d.β,α+d.α,d.space)
JacobiWeight(a::Number, b::Number, d::IntervalOrSegment) = JacobiWeight(a,b,Space(d))
JacobiWeight(a::Number, b::Number, d) = JacobiWeight(a,b,Space(d))
JacobiWeight(a::Number, b::Number) = JacobiWeight(a,b,Chebyshev())

JacobiWeight(a::Number, b::Number,s::PiecewiseSpace) = PiecewiseSpace(JacobiWeight.(a,b,components(s)))

"""
    WeightedJacobi(β, α, d::Domain = ChebyshevInterval())

The space `JacobiWeight(β,α,Jacobi(β,α,d))`.
"""
WeightedJacobi(β, α, d::Domain = ChebyshevInterval()) = JacobiWeight(β, α, Jacobi(β,α,d))


Fun(::typeof(identity), S::JacobiWeight) =
    isapproxinteger(S.β) && isapproxinteger(S.α) ? Fun(x->x,S) : Fun(identity,domain(S))

order(S::JacobiWeight{<:Ultraspherical{<:Any,D,R},D,R}) where {D,R} = order(S.space)


spacescompatible(A::JacobiWeight,B::JacobiWeight) =
    A.β ≈ B.β && A.α ≈ B.α && spacescompatible(A.space,B.space)
spacescompatible(A::JacobiWeight,B::Space{DD,RR}) where {DD<:IntervalOrSegment,RR<:Real} =
    spacescompatible(A,JacobiWeight(0,0,B))
spacescompatible(B::Space{DD,RR},A::JacobiWeight) where {DD<:IntervalOrSegment,RR<:Real} =
    spacescompatible(A,JacobiWeight(0,0,B))

transformtimes(f::Fun{JW1},g::Fun{JW2}) where {JW1<:JacobiWeight,JW2<:JacobiWeight}=
            Fun(JacobiWeight(f.space.β+g.space.β,f.space.α+g.space.α,f.space.space),
                coefficients(transformtimes(Fun(f.space.space,f.coefficients),
                                            Fun(g.space.space,g.coefficients))))
transformtimes(f::Fun{JW},g::Fun) where {JW<:JacobiWeight} =
    Fun(f.space,coefficients(transformtimes(Fun(f.space.space,f.coefficients),g)))
transformtimes(f::Fun,g::Fun{JW}) where {JW<:JacobiWeight} =
    Fun(g.space,coefficients(transformtimes(Fun(g.space.space,g.coefficients),f)))

jacobiweight(β,α,x) = -1 ≤ x ≤ 1 ? (1+x)^β*(1-x)^α : zero(x)
jacobiweight(β,α,d::Domain) = Fun(JacobiWeight(β,α,ConstantSpace(d)),[1.])
jacobiweight(β,α) = jacobiweight(β,α,ChebyshevInterval())

weight(sp::JacobiWeight,x) = jacobiweight(sp.β,sp.α,real(tocanonical(sp,x)))
dimension(sp::JacobiWeight) = dimension(sp.space)


Base.first(f::Fun{<:JacobiWeight}) = space(f).β>0 ? zero(cfstype(f)) : f(leftendpoint(domain(f)))
Base.last(f::Fun{<:JacobiWeight}) = space(f).α>0 ? zero(cfstype(f)) : f(rightendpoint(domain(f)))

setdomain(sp::JacobiWeight,d::Domain)=JacobiWeight(sp.β,sp.α,setdomain(sp.space,d))

# we assume that points avoids singularities


##TODO: paradigm for same space
function coefficients(f::AbstractVector,
        sp1::JacobiWeight{<:Any,DD},sp2::JacobiWeight{<:Any,DD}) where {DD<:IntervalOrSegment}

    β,α=sp1.β,sp1.α
    c,d=sp2.β,sp2.α

    if isapprox(c,β) && isapprox(d,α)
        # remove wrapper spaces and then convert
        coefficients(f,sp1.space,sp2.space)
    else
        # go back to default
        defaultcoefficients(f,sp1,sp2)
    end
end
function coefficients(f::AbstractVector,sp::JacobiWeight{<:Any,DD},
             S2::SubSpace{<:Any,<:Any,DD,<:Real}) where {DD<:IntervalOrSegment}
    subspace_coefficients(f,sp,S2)
end
function coefficients(f::AbstractVector,S2::SubSpace{<:Any,<:Any,DD,<:Real},
             sp::JacobiWeight{<:Any,DD}) where {DD<:IntervalOrSegment}
    subspace_coefficients(f,S2,sp)
end
#TODO: it could be possible that we want to JacobiWeight a SumSpace....
function coefficients(f::AbstractVector,
        sp::JacobiWeight{<:Any,DD},S2::SumSpace{<:Any,DD,<:Real}) where {DD<:IntervalOrSegment}
    sumspacecoefficients(f,sp,S2)
end

coefficients(f::AbstractVector,sp::JacobiWeight{<:Any,DD},S2::Space{DD,<:Real}) where {DD<:IntervalOrSegment} =
    coefficients(f,sp,JacobiWeight(0,0,S2))
coefficients(f::AbstractVector,sp::ConstantSpace{DD},ts::JacobiWeight{<:Any,DD}) where {DD<:IntervalOrSegment} =
    f.coefficients[1]*ones(ts).coefficients
coefficients(f::AbstractVector,S2::Space{DD,<:Real},sp::JacobiWeight{<:Any,DD}) where {DD<:IntervalOrSegment} =
    coefficients(f,JacobiWeight(0,0,S2),sp)


"""
`increase_jacobi_parameter(f)` multiplies by `1-x^2` on the unit interval.
`increase_jacobi_parameter(-1,f)` multiplies by `1+x` on the unit interval.
`increase_jacobi_parameter(+1,f)` multiplies by `1-x` on the unit interval.
On other domains this is accomplished by mapping to the unit interval.
"""
increase_jacobi_parameter(f) = Fun(f,JacobiWeight(f.space.β+1,f.space.α+1,space(f).space))
function increase_jacobi_parameter(s,f)
    if s==-1
        Fun(f,JacobiWeight(f.space.β+1,f.space.α,space(f).space))
    else
        Fun(f,JacobiWeight(f.space.β,f.space.α+1,space(f).space))
    end
end



function canonicalspace(S::JacobiWeight)
    if isapprox(S.β,0) && isapprox(S.α,0)
        canonicalspace(S.space)
    else
        #TODO: promote singularities?
        JacobiWeight(S.β,S.α,canonicalspace(S.space))
    end
end

function union_rule(A::ConstantSpace,B::JacobiWeight{P}) where P<:PolynomialSpace
    # we can convert to a space that contains contants provided
    # that the parameters are integers
    # when the parameters are -1 we keep them
    if isapproxinteger(B.β) && isapproxinteger(B.α)
        JacobiWeight(min(B.β,zero(B.β)),min(B.α,zero(B.α)),B.space)
    else
        NoSpace()
    end
end


## Algebra

function /(c::Number,f::Fun{<:JacobiWeight})
    g=c/Fun(space(f).space,f.coefficients)
    Fun(JacobiWeight(-f.space.β,-f.space.α,space(g)),g.coefficients)
end

function ^(f::Fun{<:JacobiWeight}, k::AbstractFloat)
    S=space(f)
    g=Fun(S.space,coefficients(f))^k
    Fun(JacobiWeight(k*S.β,k*S.α,space(g)),coefficients(g))
end

function *(f::Fun{<:JacobiWeight}, g::Fun{<:JacobiWeight})
    @assert domainscompatible(f,g)
    fβ,fα=f.space.β,f.space.α
    gβ,gα=g.space.β,g.space.α
    m=(Fun(space(f).space,f.coefficients).*Fun(space(g).space,g.coefficients))
    if isapprox(fβ+gβ,0) && isapprox(fα+gα,0)
        m
    else
        Fun(JacobiWeight(fβ+gβ,fα+gα,space(m)),m.coefficients)
    end
end


/(f::Fun{<:JacobiWeight}, g::Fun{<:JacobiWeight}) = f*(1/g)

# O(min(m,n)) Ultraspherical conjugated inner product

function conjugatedinnerproduct(sp::Ultraspherical,u::AbstractVector{S},v::AbstractVector{V}) where {S,V}
    λ=order(sp)
    if λ==1
        mn = min(length(u),length(v))
        if mn > 0
            return dotu(u[1:mn],v[1:mn])*π/2
        else
            return zero(promote_type(eltype(u),eltype(v)))
        end
    else
        T,mn = promote_type(S,V),min(length(u),length(v))
        if mn > 1
            wi = sqrt(strictconvert(T,π))*gamma(λ+one(T)/2)/gamma(λ+one(T))
            ret = u[1]*wi*v[1]
            for i=2:mn
              wi *= (i-2one(T)+2λ)/(i-one(T)+λ)*(i-2one(T)+λ)/(i-one(T))
              ret += u[i]*wi*v[i]
            end
            return ret
        elseif mn > 0
            wi = sqrt(strictconvert(T,π))*gamma(λ+one(T)/2)/gamma(λ+one(T))
            return u[1]*wi*v[1]
        else
            return zero(promote_type(eltype(u),eltype(v)))
        end
    end
end

function conjugatedinnerproduct(::Chebyshev,u::AbstractVector,v::AbstractVector)
    mn = min(length(u),length(v))
    if mn > 1
        return (2u[1]*v[1]+dotu(u[2:mn],v[2:mn]))*π/2
    elseif mn > 0
        return u[1]*v[1]*π
    else
        return zero(promote_type(eltype(u),eltype(v)))
    end
end

function _bilinearformU(lfn, d, λ, f, g)
    lfn(d)/2*conjugatedinnerproduct(Ultraspherical(λ,d), coefficients(g), coefficients(g))
end

function bilinearform(f::Fun{<:JacobiWeight{U,D,R}}, g::Fun{U}) where {D,R,U<:Ultraspherical{<:Any,D,R}}
    d = domain(f)
    @assert d == domain(g)
    λ = order(space(f).space)
    if isapproxminhalf(f.space.β - λ) && isapproxminhalf(f.space.α - λ) && order(space(g)) == λ
        return _bilinearformU(complexlength, d, λ, f, g)
    else
        return defaultbilinearform(f,g)
    end
end

function bilinearform(f::Fun{U}, g::Fun{<:JacobiWeight{U,D,R}}) where {D,R,U<:Ultraspherical{<:Any,D,R}}
    d = domain(f)
    @assert d == domain(g)
    λ = order(space(f))
    if isapproxminhalf(g.space.β - λ) && isapproxminhalf(g.space.α - λ) && order(space(g).space) == λ
        return _bilinearformU(complexlength, d, λ, f, g)
    else
        return defaultbilinearform(f,g)
    end
end

function bilinearform(f::Fun{<:JacobiWeight{U,D,R}},
        g::Fun{<:JacobiWeight{U,D,R}}) where {D,R,U<:Ultraspherical{<:Any,D,R}}

    d = domain(f)
    @assert d == domain(g)
    λ = order(space(f).space)
    if isapproxminhalf(f.space.β+g.space.β-λ) && isapproxminhalf(f.space.α+g.space.α-λ) &&
            order(space(g).space) == λ
        return _bilinearformU(complexlength, d, λ, f, g)
    else
        return defaultbilinearform(f,g)
    end
end

function linebilinearform(f::Fun{<:JacobiWeight{U,D,R}},
        g::Fun{U}) where {D,R,U<:Ultraspherical{<:Any,D,R}}

    d = domain(f)
    @assert d == domain(g)
    λ = order(space(f).space)
    if isapproxminhalf(f.space.β - λ) && isapproxminhalf(f.space.α - λ) && order(space(g)) == λ
        return _bilinearformU(arclength, d, λ, f, g)
    else
        return defaultlinebilinearform(f,g)
    end
end

function linebilinearform(f::Fun{U},
        g::Fun{<:JacobiWeight{U,D,R}}) where {D,R,U<:Ultraspherical{<:Any,D,R}}

    d = domain(f)
    @assert d == domain(g)
    λ = order(space(f))
    if isapproxminhalf(g.space.β-λ) && isapproxminhalf(g.space.α-λ) && order(space(g).space) == λ
        return _bilinearformU(arclength, d, λ, f, g)
    else
        return defaultlinebilinearform(f,g)
    end
end

function linebilinearform(f::Fun{<:JacobiWeight{U,D,R}},
        g::Fun{<:JacobiWeight{U,D,R}}) where {D,R,U<:Ultraspherical{<:Any,D,R}}

    d = domain(f)
    @assert d == domain(g)
    λ = order(space(f).space)
    if isapproxminhalf(f.space.β+g.space.β-λ) && isapproxminhalf(f.space.α+g.space.α-λ) &&
            order(space(g).space) == λ
        return _bilinearformU(arclength, d, λ, f, g)
    else
        return defaultlinebilinearform(f,g)
    end
end


function bilinearform(f::Fun{<:JacobiWeight{J,<:IntervalOrSegment}}, g::Fun{J}) where {J<:Jacobi}
    @assert domain(f) == domain(g)
    if f.space.β == f.space.space.a == g.space.a && f.space.α == f.space.space.b == g.space.b
        return complexlength(domain(f))/2*conjugatedinnerproduct(g.space,f.coefficients,g.coefficients)
    else
        return defaultbilinearform(f,g)
    end
end

function bilinearform(f::Fun{J},
                      g::Fun{<:JacobiWeight{J,<:IntervalOrSegment}}) where {J<:Jacobi}
    @assert domain(f) == domain(g)
    if g.space.β == g.space.space.a == f.space.a && g.space.α == g.space.space.b == f.space.b
        return complexlength(domain(f))/2*conjugatedinnerproduct(f.space,f.coefficients,g.coefficients)
    else
        return defaultbilinearform(f,g)
    end
end

function bilinearform(f::Fun{JW}, g::Fun{JW}) where {J<:Jacobi,DD<:IntervalOrSegment,JW<:JacobiWeight{J,DD}}
    @assert domain(f) == domain(g)
    if f.space.β + g.space.β == f.space.space.a == g.space.space.a && f.space.α + g.space.α == f.space.space.b == g.space.space.b
        return complexlength(domain(f))/2*conjugatedinnerproduct(f.space.space,f.coefficients,g.coefficients)
    else
        return defaultbilinearform(f,g)
    end
end


function linebilinearform(f::Fun{<:JacobiWeight{J,<:IntervalOrSegment}}, g::Fun{J}) where {J<:Jacobi}
    @assert domain(f) == domain(g)
    if f.space.β == f.space.space.a == g.space.a && f.space.α == f.space.space.b == g.space.b
        return arclength(domain(f))/2*conjugatedinnerproduct(g.space,f.coefficients,g.coefficients)
    else
        return defaultlinebilinearform(f,g)
    end
end

function linebilinearform(f::Fun{J}, g::Fun{<:JacobiWeight{J,<:IntervalOrSegment}}) where {J<:Jacobi}
    @assert domain(f) == domain(g)
    if g.space.β == g.space.space.a == f.space.a && g.space.α == g.space.space.b == f.space.b
        return arclength(domain(f))/2*conjugatedinnerproduct(f.space,f.coefficients,g.coefficients)
    else
        return defaultlinebilinearform(f,g)
    end
end

function linebilinearform(f::Fun{JW}, g::Fun{JW}) where {
            J<:Jacobi,DD<:IntervalOrSegment,JW<:JacobiWeight{J,DD}}
    @assert domain(f) == domain(g)
    if f.space.β + g.space.β == f.space.space.a == g.space.space.a && f.space.α + g.space.α == f.space.space.b == g.space.space.b
        return arclength(domain(f))/2*conjugatedinnerproduct(f.space.space,f.coefficients,g.coefficients)
    else
        return defaultlinebilinearform(f,g)
    end
end


function Derivative(S::WeightedJacobi{<:IntervalOrSegment})
    if S.β>0 && S.β>0 && S.β==S.space.b && S.α==S.space.a
        ConcreteDerivative(S,1)
    else
        jacobiweightDerivative(S)
    end
end

bandwidths(D::ConcreteDerivative{<:WeightedJacobi{<:IntervalOrSegment}}) = 1,0
rangespace(D::ConcreteDerivative{<:WeightedJacobi{<:IntervalOrSegment}}) =
    WeightedJacobi(domainspace(D).β-1,domainspace(D).α-1,domain(D))

getindex(D::ConcreteDerivative{<:WeightedJacobi{<:IntervalOrSegment}}, k::Integer, j::Integer) =
    j==k-1 ? eltype(D)(-4(k-1)./complexlength(domain(D))) : zero(eltype(D))




for (Func,Len,Sum) in ((:DefiniteIntegral,:complexlength,:sum),(:DefiniteLineIntegral,:arclength,:linesum))
    ConcFunc = Symbol(:Concrete, Func)

    @eval begin
        function getindex(Σ::$ConcFunc{<:JacobiWeight{<:Jacobi{D,R},D,R},T}, k::Integer) where {D<:IntervalOrSegment,R,T}
            dsp = domainspace(Σ)

            if dsp.β == dsp.space.b && dsp.α == dsp.space.a
                # TODO: copy and paste
                k == 1 ? strictconvert(T,$Sum(Fun(dsp,[one(T)]))) : zero(T)
            else
                strictconvert(T,$Sum(Fun(dsp,[zeros(T,k-1);1])))
            end
        end

        function bandwidths(Σ::$ConcFunc{<:JacobiWeight{<:Jacobi{D,R},D,R}}) where {D<:IntervalOrSegment,R}
            β,α = domainspace(Σ).β,domainspace(Σ).α
            if domainspace(Σ).β == domainspace(Σ).space.b && domainspace(Σ).α == domainspace(Σ).space.a
                0,0  # first entry
            else
                0,∞
            end
        end
    end
end


## <: IntervalOrSegment avoids a julia bug
function Multiplication(f::Fun{<:JacobiWeight{<:ConstantSpace,<:IntervalOrSegmentDomain}}, S::Jacobi)
    # this implements (1+x)*P and (1-x)*P special case
    # see DLMF (18.9.6)
    d=domain(f)
    Sf = space(f)
    if ((Sf.β==1 && Sf.α==0 && S.b >0) ||
                        (Sf.β==0 && Sf.α==1 && S.a >0))
        ConcreteMultiplication(f,S)
    elseif isapproxinteger(Sf.β) && Sf.β ≥ 1 && S.b >0
        # decrement β and multiply again
        M1 = ConcreteMultiplication(f.coefficients[1]*jacobiweight(1,0,d),S)
        M1_out = Multiplication(jacobiweight(Sf.β-1,Sf.α,d), rangespace(M1)) * M1
        MultiplicationWrapper(f, M1_out)
    elseif isapproxinteger(Sf.α) && Sf.α ≥ 1 && S.a >0
        # decrement α and multiply again
        M2 = ConcreteMultiplication(f.coefficients[1]*jacobiweight(0,1,d),S)
        M2_out = Multiplication(jacobiweight(Sf.β,Sf.α-1,d), rangespace(M2)) * M2
        MultiplicationWrapper(f, M2_out)
    else
        # default JacobiWeight
        M = Multiplication(Fun(Sf.space, f.coefficients), S)
        rsp = JacobiWeight(Sf.β, Sf.α, rangespace(M))
        MultiplicationWrapper(f, SpaceOperator(M,S,rsp))
    end
end

Multiplication(f::Fun{<:JacobiWeight{<:ConstantSpace,<:IntervalOrSegmentDomain}},
               S::Union{Ultraspherical,Chebyshev}) =
    MultiplicationWrapper(f,Multiplication(f,Jacobi(S))*Conversion(S,Jacobi(S)))

function rangespace(M::ConcreteMultiplication{<:
                        JacobiWeight{<:ConstantSpace,<:IntervalOrSegmentDomain},<:Jacobi})
    S=domainspace(M)
    if space(M.f).β==1
        # multiply by (1+x)
        Jacobi(S.b-1,S.a,domain(S))
    elseif space(M.f).α == 1
        # multiply by (1-x)
        Jacobi(S.b,S.a-1,domain(S))
    else
        error("Not implemented")
    end
end

bandwidths(::ConcreteMultiplication{<:
                JacobiWeight{<:ConstantSpace,<:IntervalOrSegmentDomain},<:Jacobi}) = 1,0


function getindex(M::ConcreteMultiplication{
            <:JacobiWeight{<:ConstantSpace,<:IntervalOrSegmentDomain},<:Jacobi},
                k::Integer, j::Integer)
    @assert ncoefficients(M.f)==1
    a,b=domainspace(M).a,domainspace(M).b
    c=M.f.coefficients[1]
    if space(M.f).β==1
        @assert space(M.f).α==0
        # multiply by (1+x)
        if j==k
            c*2(k+b-1)/(2k+a+b-1)
        elseif k > 1 && j==k-1
            c*(2k-2)/(2k+a+b-3)
        else
            zero(eltype(M))
        end
    elseif space(M.f).α == 1
        @assert space(M.f).β==0
        # multiply by (1-x)
        if j==k
            c*2(k+a-1)/(2k+a+b-1)
        elseif k > 1 && j==k-1
            -c*(2k-2)/(2k+a+b-3)
        else
            zero(eltype(M))
        end
    else
        error("Not implemented")
    end
end


# We can exploit the special multiplication to construct a Conversion


for FUNC in (:maxspace_rule,:union_rule,:hasconversion)
    @eval function $FUNC(A::WeightedJacobi{<:IntervalOrSegment}, B::Jacobi)
        if A.β==A.α+1 && A.space.b>0
            $FUNC(Jacobi(A.space.b-1,A.space.a,domain(A)),B)
        elseif A.α==A.β+1 && A.space.a>0
            $FUNC(Jacobi(A.space.b,A.space.a-1,domain(A)),B)
        else
            $FUNC(A,JacobiWeight(0.,0.,B))
        end
    end
end

function show(io::IO,s::JacobiWeight)
    d=domain(s)
    #TODO: Get shift and weights right
    sym = domain(s) == canonicaldomain(s) ? "x" : "𝑪($(domain(s)), x)"
    if s.α==s.β
        print(io,"(1-$sym^2)^", s.α)
    elseif s.β==0
        print(io,"(1-$sym)^", s.α)
    elseif s.α==0
        print(io,"(1+$sym)^", s.β)
    else
        print(io,"(1+$sym)^", s.β, " * (1-$sym)^", s.α)
    end
    print(io, " * ")
    show(io, s.space)
end

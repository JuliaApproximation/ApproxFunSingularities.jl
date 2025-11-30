module AFSTests

using ApproxFunSingularities
using ParallelTestRunner

const init_code = quote
    using ApproxFunBase
    using ApproxFunBase: HeavisideSpace, PointSpace, ArraySpace, DiracSpace, PiecewiseSegment,
                            UnionDomain, resizedata!, CachedOperator, RaggedMatrix,
                            Block, ∞, BandedBlockBandedMatrix, NoSpace, ConcreteMultiplication,
                            MultiplicationWrapper
    using ApproxFunBase.TestUtils: testbandedoperator, testtransforms, testfunctional,
                            testbandedblockbandedoperator
    using ApproxFunOrthogonalPolynomials
    using ApproxFunOrthogonalPolynomials: order
    using ApproxFunSingularities
    using HalfIntegers
    using IntervalSets
    using LinearAlgebra
    using OddEvenIntegers
    using SpecialFunctions
    using Test
end

args = parse_args(ARGS)

if filter_tests!(testsuite, args)
    delete!(testsuite, "testutils.jl")
end

runtests(ApproxFunSingularities, ARGS; init_code, testsuite)

end # module

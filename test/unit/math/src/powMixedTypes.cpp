/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera, Bernhard Manfred Gruber,
 *                Sergei Bastrakov, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include "Defines.hpp"

#include <aV2P/math/Traits.hpp>
#include <aV2P/test/KernelExecutionFixture.hpp>
#include <aV2P/test/acc/TestAccs.hpp>
#include <aV2P/test/queue/Queue.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <type_traits>

//! Convert the given real or complex input type to the given real or complex output type avoiding warnings.
//! This conversion is surprisingly tricky to do in a way that no compiler complains.
//! In principle it could be accomplished by a constexpr function, but in practice that turned out not possible.
//! The general implementation does direct initialization, works for pairs of types supporting it.
template<typename TInput, typename TOutput, typename TSfinae = void>
struct Convert
{
    AV2P_FN_ACC auto operator()(TInput const arg) const
    {
        return TOutput{arg};
    }
};

//! Specialization for real -> real conversion
template<typename TInput, typename TOutput>
struct Convert<TInput, TOutput, std::enable_if_t<std::is_floating_point_v<TOutput>>>
{
    AV2P_FN_ACC auto operator()(TInput const arg) const
    {
        return static_cast<TOutput>(arg);
    }
};

//! Specialization for real -> complex conversion
template<typename TInput, typename TOutputValueType>
struct Convert<TInput, aV2P::Complex<TOutputValueType>, std::enable_if_t<std::is_floating_point_v<TInput>>>
{
    AV2P_FN_ACC auto operator()(TInput const arg) const
    {
        return aV2P::Complex<TOutputValueType>{static_cast<TOutputValueType>(arg)};
    }
};

template<typename TExpected>
struct PowMixedTypesTestKernel
{
    AV2P_NO_HOST_ACC_WARNING
    template<typename TAcc, typename TArg1, typename TArg2>
    AV2P_FN_ACC auto operator()(TAcc const& acc, bool* success, TArg1 const arg1, TArg2 const arg2) const -> void
    {
        auto expected = aV2P::math::pow(acc, Convert<TArg1, TExpected>{}(arg1), Convert<TArg2, TExpected>{}(arg2));
        auto actual = aV2P::math::pow(acc, arg1, arg2);
        AV2P_CHECK(*success, mathtest::almost_equal(acc, expected, actual, 1));
    }
};

using TestAccs = aV2P::test::EnabledAccs<aV2P::DimInt<1u>, std::size_t>;

TEMPLATE_LIST_TEST_CASE("powMixedTypes", "[powMixedTypes]", TestAccs)
{
    using Acc = TestType;
    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;

    aV2P::test::KernelExecutionFixture<Acc> fixture(aV2P::Vec<Dim, Idx>::ones());

    PowMixedTypesTestKernel<float> kernelFloat;
    PowMixedTypesTestKernel<double> kernelDouble;
    PowMixedTypesTestKernel<aV2P::Complex<float>> kernelComplexFloat;
    PowMixedTypesTestKernel<aV2P::Complex<double>> kernelComplexDouble;

    float const floatArg = 0.35f;
    double const doubleArg = 0.24;
    aV2P::Complex<float> floatComplexArg{0.35f, -0.24f};
    aV2P::Complex<double> doubleComplexArg{0.35, -0.24};

    // all combinations of pow(real, real)
    REQUIRE(fixture(kernelFloat, floatArg, floatArg));
    REQUIRE(fixture(kernelDouble, floatArg, doubleArg));
    REQUIRE(fixture(kernelDouble, doubleArg, floatArg));
    REQUIRE(fixture(kernelDouble, doubleArg, doubleArg));

    // all combinations of pow(real, complex)
    REQUIRE(fixture(kernelComplexFloat, floatArg, floatComplexArg));
    REQUIRE(fixture(kernelComplexDouble, floatArg, doubleComplexArg));
    REQUIRE(fixture(kernelComplexDouble, doubleArg, floatComplexArg));
    REQUIRE(fixture(kernelComplexDouble, doubleArg, doubleComplexArg));

    // all combinations of pow(complex, real)
    REQUIRE(fixture(kernelComplexFloat, floatComplexArg, floatArg));
    REQUIRE(fixture(kernelComplexDouble, floatComplexArg, doubleArg));
    REQUIRE(fixture(kernelComplexDouble, doubleComplexArg, floatArg));
    REQUIRE(fixture(kernelComplexDouble, doubleComplexArg, doubleArg));

    // all combinations of pow(complex, complex)
    REQUIRE(fixture(kernelComplexFloat, floatComplexArg, floatComplexArg));
    REQUIRE(fixture(kernelComplexDouble, floatComplexArg, doubleComplexArg));
    REQUIRE(fixture(kernelComplexDouble, doubleComplexArg, floatComplexArg));
    REQUIRE(fixture(kernelComplexDouble, doubleComplexArg, doubleComplexArg));
}

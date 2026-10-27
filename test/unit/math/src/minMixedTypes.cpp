/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/math/MathUniformCudaHipBuiltIn.hpp>
#include <aV2P/math/Traits.hpp>
#include <aV2P/test/Check.hpp>
#include <aV2P/test/KernelExecutionFixture.hpp>
#include <aV2P/test/acc/TestAccs.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <type_traits>

namespace min_fix_test
{
    //! Aggregate type without any alpaka math support of its own.
    template<typename T>
    struct Pair
    {
        T first;
        T second;
    };
} // namespace min_fix_test

//! Replicates a client specialization with an unconditional `void` SFINAE argument.
//! Before the fix this overlapped with the accelerator facade's generic `Min`.
namespace aV2P::math::trait
{
    template<typename TCtx, typename T>
    struct Min<TCtx, min_fix_test::Pair<T>, min_fix_test::Pair<T>, void>
    {
        AV2P_FN_HOST_ACC auto operator()(
            TCtx const& ctx,
            min_fix_test::Pair<T> const& lhs,
            min_fix_test::Pair<T> const& rhs) const -> min_fix_test::Pair<T>
        {
            return {aV2P::math::min(ctx, lhs.first, rhs.first), aV2P::math::min(ctx, lhs.second, rhs.second)};
        }
    };
} // namespace aV2P::math::trait

namespace
{
    struct MinScalarKernel
    {
        template<typename TAcc, typename Tx, typename Ty>
        AV2P_FN_ACC auto operator()(TAcc const& acc, bool* success, Tx const& x, Ty const& y) const -> void
        {
            auto const result = aV2P::math::min(acc, x, y);
            double const expected
                = static_cast<double>(x) < static_cast<double>(y) ? static_cast<double>(x) : static_cast<double>(y);
            double const actual = static_cast<double>(result);
            AV2P_CHECK(*success, !(actual < expected) && !(expected < actual));
        }
    };

    struct MinPairKernel
    {
        template<typename TAcc>
        AV2P_FN_ACC auto operator()(TAcc const& acc, bool* success, float x, float y) const -> void
        {
            using Pair = min_fix_test::Pair<float>;
            Pair const result = aV2P::math::min(acc, Pair{x, 1.0f}, Pair{y, 2.0f});
            float const expected = x < y ? x : y;
            AV2P_CHECK(*success, !(result.first < expected) && !(expected < result.first));
            AV2P_CHECK(*success, !(result.second < 1.0f) && !(1.0f < result.second));
        }
    };
} // namespace

using TestAccs = aV2P::test::EnabledAccs<aV2P::DimInt<1u>, std::size_t>;

TEMPLATE_LIST_TEST_CASE("mathMinScalarOnAccelerator", "[math][min]", TestAccs)
{
    using Acc = TestType;

    aV2P::test::KernelExecutionFixture<Acc> fixture(aV2P::Vec<aV2P::Dim<Acc>, aV2P::Idx<Acc>>::ones());
    MinScalarKernel const kernel;

    REQUIRE(fixture(kernel, 1.5f, 2.5f));
    REQUIRE(fixture(kernel, 1.5, 2.5));
    REQUIRE(fixture(kernel, 1.5f, 2.5));
    REQUIRE(fixture(kernel, 1.5, 2.5f));
    REQUIRE(fixture(kernel, 2, 1));
}

TEMPLATE_LIST_TEST_CASE("mathMinClientSpecializationIsUnambiguous", "[math][min]", TestAccs)
{
    using Acc = TestType;

    aV2P::test::KernelExecutionFixture<Acc> fixture(aV2P::Vec<aV2P::Dim<Acc>, aV2P::Idx<Acc>>::ones());
    REQUIRE(fixture(MinPairKernel{}, 1.5f, 2.5f));
}

TEST_CASE("mathMinCudaBuiltInMathContext", "[math][min]")
{
    auto const f = aV2P::math::min(aV2P::math::MinUniformCudaHipBuiltIn{}, 1.5f, 2.5f);
    STATIC_REQUIRE(std::is_same_v<std::decay_t<decltype(f)>, float>);
    REQUIRE(f == Catch::Approx(1.5f));

    auto const d = aV2P::math::min(aV2P::math::MinUniformCudaHipBuiltIn{}, 1.5f, 2.5);
    STATIC_REQUIRE(std::is_same_v<std::decay_t<decltype(d)>, double>);
    REQUIRE(d == Catch::Approx(1.5));

    auto const i = aV2P::math::min(aV2P::math::MinUniformCudaHipBuiltIn{}, 2, 1);
    STATIC_REQUIRE(std::is_same_v<std::decay_t<decltype(i)>, int>);
    REQUIRE(i == 1);
}

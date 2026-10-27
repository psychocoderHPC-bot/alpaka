/* Copyright 2022 Axel Huebl, Benjamin Worpitz, René Widera, Bernhard Manfred Gruber, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/kernel/Traits.hpp>
#include <aV2P/meta/ForEachType.hpp>
#include <aV2P/test/KernelExecutionFixture.hpp>
#include <aV2P/test/acc/TestAccs.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <type_traits>

template<typename T>
class KernelFuntionObjectTemplate
{
public:
    AV2P_NO_HOST_ACC_WARNING
    template<typename TAcc>
    AV2P_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        AV2P_CHECK(
            *success,
            static_cast<aV2P::Idx<TAcc>>(1) == (aV2P::getWorkDiv<aV2P::Grid, aV2P::Threads>(acc)).prod());

        static_assert(std::is_same_v<std::int32_t, T>, "Incorrect additional kernel template parameter type!");
    }
};

TEMPLATE_LIST_TEST_CASE("kernelFuntionObjectTemplate", "[kernel]", aV2P::test::TestAccs)
{
    using Acc = TestType;
    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;

    aV2P::test::KernelExecutionFixture<Acc> fixture(aV2P::Vec<Dim, Idx>::ones());

    KernelFuntionObjectTemplate<std::int32_t> kernel;

    REQUIRE(fixture(kernel));
}

class KernelInvocationWithAdditionalTemplate
{
public:
    AV2P_NO_HOST_ACC_WARNING
    template<typename TAcc, typename T>
    AV2P_FN_ACC auto operator()(TAcc const& acc, bool* success, T const&) const -> void
    {
        AV2P_CHECK(
            *success,
            static_cast<aV2P::Idx<TAcc>>(1) == (aV2P::getWorkDiv<aV2P::Grid, aV2P::Threads>(acc)).prod());

        static_assert(std::is_same_v<std::int32_t, T>, "Incorrect additional kernel template parameter type!");
    }
};

TEMPLATE_LIST_TEST_CASE("kernelFuntionObjectExtraTemplate", "[kernel]", aV2P::test::TestAccs)
{
    using Acc = TestType;
    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;

    aV2P::test::KernelExecutionFixture<Acc> fixture(aV2P::Vec<Dim, Idx>::ones());

    KernelInvocationWithAdditionalTemplate kernel;

    REQUIRE(fixture(kernel, std::int32_t()));
}

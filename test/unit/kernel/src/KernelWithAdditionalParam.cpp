/* Copyright 2022 Axel Huebl, Benjamin Worpitz, René Widera, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/kernel/Traits.hpp>
#include <aV2P/meta/ForEachType.hpp>
#include <aV2P/test/KernelExecutionFixture.hpp>
#include <aV2P/test/acc/TestAccs.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

class KernelWithAdditionalParamByValue
{
public:
    AV2P_NO_HOST_ACC_WARNING
    template<typename TAcc>
    AV2P_FN_ACC auto operator()(TAcc const& /* acc */, bool* success, std::int32_t val) const -> void
    {
        AV2P_CHECK(*success, 42 == val);
    }
};

TEMPLATE_LIST_TEST_CASE("KernelWithAdditionalParamByValue", "[kernel]", aV2P::test::TestAccs)
{
    using Acc = TestType;
    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;

    aV2P::test::KernelExecutionFixture<Acc> fixture(aV2P::Vec<Dim, Idx>::ones());

    KernelWithAdditionalParamByValue kernel;

    REQUIRE(fixture(kernel, 42));
}

class KernelWithAdditionalParamByConstRef
{
public:
    AV2P_NO_HOST_ACC_WARNING
    template<typename TAcc>
    AV2P_FN_ACC auto operator()(TAcc const& /* acc */, bool* success, std::int32_t const& val) const -> void
    {
        AV2P_CHECK(*success, 42 == val);
    }
};

TEMPLATE_LIST_TEST_CASE("KernelWithAdditionalParamByConstRef", "[kernel]", aV2P::test::TestAccs)
{
    using Acc = TestType;
    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;

    aV2P::test::KernelExecutionFixture<Acc> fixture(aV2P::Vec<Dim, Idx>::ones());

    KernelWithAdditionalParamByConstRef kernel;

    REQUIRE(fixture(kernel, 42));
}

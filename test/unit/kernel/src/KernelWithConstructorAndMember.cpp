/* Copyright 2022 Axel Huebl, Benjamin Worpitz, René Widera, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/kernel/Traits.hpp>
#include <aV2P/meta/ForEachType.hpp>
#include <aV2P/test/KernelExecutionFixture.hpp>
#include <aV2P/test/acc/TestAccs.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

class KernelWithConstructorAndMember
{
public:
    AV2P_FN_HOST KernelWithConstructorAndMember(std::int32_t const val = 42) : m_val(val)
    {
    }

    AV2P_NO_HOST_ACC_WARNING
    template<typename TAcc>
    AV2P_FN_ACC auto operator()(TAcc const& /* acc */, bool* success) const -> void
    {
        AV2P_CHECK(*success, 42 == m_val);
    }

private:
    std::int32_t m_val;
};

TEMPLATE_LIST_TEST_CASE("kernelWithConstructorAndMember", "[kernel]", aV2P::test::TestAccs)
{
    using Acc = TestType;
    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;

    aV2P::test::KernelExecutionFixture<Acc> fixture(aV2P::Vec<Dim, Idx>::ones());

    KernelWithConstructorAndMember kernel(42);

    REQUIRE(fixture(kernel));
}

TEMPLATE_LIST_TEST_CASE("kernelWithConstructorDefaultParamAndMember", "[kernel]", aV2P::test::TestAccs)
{
    using Acc = TestType;
    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;

    aV2P::test::KernelExecutionFixture<Acc> fixture(aV2P::Vec<Dim, Idx>::ones());

    KernelWithConstructorAndMember kernel;

    REQUIRE(fixture(kernel));
}

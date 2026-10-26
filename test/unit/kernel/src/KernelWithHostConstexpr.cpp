/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/kernel/Traits.hpp>
#include <aV2P/meta/ForEachType.hpp>
#include <aV2P/test/KernelExecutionFixture.hpp>
#include <aV2P/test/acc/TestAccs.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <limits>

class KernelWithHostConstexpr
{
public:
    AV2P_NO_HOST_ACC_WARNING
    template<typename TAcc>
    AV2P_FN_ACC auto operator()(TAcc const& /* acc */, bool* success) const -> void
    {
#if AV2P_COMP_MSVC
#    pragma warning(push)
#    pragma warning(disable : 4127) // warning C4127: conditional expression is constant
#endif

        constexpr auto max = std::numeric_limits<std::uint32_t>::max();

        AV2P_CHECK(*success, 0 != max);
#if AV2P_COMP_MSVC
#    pragma warning(pop)
#endif
    }
};

TEMPLATE_LIST_TEST_CASE("kernelWithHostConstexpr", "[kernel]", aV2P::test::TestAccs)
{
    using Acc = TestType;
    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;

    aV2P::test::KernelExecutionFixture<Acc> fixture(aV2P::Vec<Dim, Idx>::ones());

    KernelWithHostConstexpr kernel;

    REQUIRE(fixture(kernel));
}

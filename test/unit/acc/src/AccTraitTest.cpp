/* Copyright 2024 Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/acc/Traits.hpp>
#include <aV2P/test/acc/TestAccs.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

TEMPLATE_LIST_TEST_CASE("isSingleThreadAcc", "[acc]", aV2P::test::TestAccs)
{
    using Acc = TestType;

    // Check that both traits are defined, and that only one is true.
    REQUIRE(aV2P::isSingleThreadAcc<Acc> != aV2P::isMultiThreadAcc<Acc>);

    auto const platform = aV2P::Platform<Acc>{};
    auto const dev = aV2P::getDevByIdx(platform, 0);
    auto const devProps = aV2P::getAccDevProps<Acc>(dev);

    // Compare the runtime properties with the compile time trait.
    INFO("Accelerator: " << aV2P::core::demangled<Acc>);
    if constexpr(aV2P::isSingleThreadAcc<Acc>)
    {
        // Require a single thread per block.
        REQUIRE(devProps.m_blockThreadCountMax == 1);
    }
    else
    {
        // Assume multiple threads per block, but allow a single thread per block.
        // For example, the AccCpuOmp2Threads accelerator may report a single thread on a single core system.
        REQUIRE(devProps.m_blockThreadCountMax >= 1);
    }
}

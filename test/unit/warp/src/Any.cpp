/* Copyright 2023 Sergei Bastrakov, Bernhard Manfred Gruber, Jan Stephan, Andrea Bocci, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#include "SyclWarpSupport.hpp"

#include <aV2P/test/KernelExecutionFixture.hpp>
#include <aV2P/test/acc/TestAccs.hpp>
#include <aV2P/test/queue/Queue.hpp>
#include <aV2P/warp/Traits.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>

struct AnySingleThreadWarpTestKernel
{
    AV2P_NO_HOST_ACC_WARNING
    template<typename TAcc>
    AV2P_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        if constexpr(aV2P::Dim<TAcc>::value > 0)
            AV2P_CHECK(*success, aV2P::warp::getSize(acc) == 1);
        AV2P_CHECK(*success, aV2P::warp::any(acc, 42) != 0);
        AV2P_CHECK(*success, aV2P::warp::any(acc, 0) == 0);
    }
};

template<std::uint32_t TWarpSize>
struct AnyMultipleThreadWarpTestKernel
{
    AV2P_NO_HOST_ACC_WARNING
    template<typename TAcc>
    AV2P_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        std::int32_t const warpExtent = aV2P::warp::getSize(acc);
        AV2P_CHECK(*success, warpExtent > 1);

        AV2P_CHECK(*success, aV2P::warp::any(acc, 0) == 0);
        AV2P_CHECK(*success, aV2P::warp::any(acc, 42) != 0);

        // Test relies on having a single warp per thread block
        auto const blockExtent = aV2P::getWorkDiv<aV2P::Block, aV2P::Threads>(acc);
        AV2P_CHECK(*success, static_cast<std::int32_t>(blockExtent.prod()) == warpExtent);
        auto const localThreadIdx = aV2P::getIdx<aV2P::Block, aV2P::Threads>(acc);
        auto const threadIdxInWarp = static_cast<std::int32_t>(aV2P::mapIdx<1u>(localThreadIdx, blockExtent)[0]);

        // Some threads quit the kernel to test that the warp operations
        // properly operate on the active threads only
        if(threadIdxInWarp % 2)
            return;

        for(auto idx = 0; idx < warpExtent; idx++)
        {
            AV2P_CHECK(*success, aV2P::warp::any(acc, threadIdxInWarp == idx ? 0 : 1) == 1);
            std::int32_t const expected = idx % 2 ? 0 : 1;
            AV2P_CHECK(*success, aV2P::warp::any(acc, threadIdxInWarp == idx ? 1 : 0) == expected);
        }
    }
};

template<std::uint32_t TWarpSize, typename TAcc>
struct aV2P::trait::WarpSize<AnyMultipleThreadWarpTestKernel<TWarpSize>, TAcc>
    : std::integral_constant<std::uint32_t, TWarpSize>
{
};

TEMPLATE_LIST_TEST_CASE("any", "[warp]", aV2P::test::TestAccs)
{
    using Acc = TestType;

    if constexpr(constexpr char const* reason = aV2P::test::warpUnsupportedReason<Acc>(); reason != nullptr)
    {
        WARN("Test disabled: " << reason);
    }
    else
    {
        using Dim = aV2P::Dim<Acc>;
        using Idx = aV2P::Idx<Acc>;

        auto const platform = aV2P::Platform<Acc>{};
        auto const dev = aV2P::getDevByIdx(platform, 0);
        auto const warpExtents = aV2P::getWarpSizes(dev);
        for(auto const warpExtent : warpExtents)
        {
            auto const scalar = Dim::value == 0 || warpExtent == 1;
            if(scalar)
            {
                aV2P::test::KernelExecutionFixture<Acc> fixture(aV2P::Vec<Dim, Idx>::all(4));
                REQUIRE(fixture(AnySingleThreadWarpTestKernel{}));
            }
            else
            {
                using ExecutionFixture = aV2P::test::KernelExecutionFixture<Acc>;
                auto const gridBlockExtent = aV2P::Vec<Dim, Idx>::all(2);
                // Enforce one warp per thread block
                auto blockThreadExtent = aV2P::Vec<Dim, Idx>::ones();
                // Put the whole warp along the innermost (fastest) dimension: the SYCL CPU device forms sub-groups
                // only along it, while GPUs form warps from the linearised thread index.
                constexpr auto innermost = Dim::value > 0 ? Dim::value - 1 : 0;
                blockThreadExtent[innermost] = static_cast<Idx>(warpExtent);
                auto const threadElementExtent = aV2P::Vec<Dim, Idx>::ones();
                auto workDiv =
                    typename ExecutionFixture::WorkDiv{gridBlockExtent, blockThreadExtent, threadElementExtent};
                auto fixture = ExecutionFixture{workDiv};
                if(warpExtent == 4)
                {
                    REQUIRE(fixture(AnyMultipleThreadWarpTestKernel<4>{}));
                }
                else if(warpExtent == 8)
                {
                    REQUIRE(fixture(AnyMultipleThreadWarpTestKernel<8>{}));
                }
                else if(warpExtent == 16)
                {
                    REQUIRE(fixture(AnyMultipleThreadWarpTestKernel<16>{}));
                }
                else if(warpExtent == 32)
                {
                    REQUIRE(fixture(AnyMultipleThreadWarpTestKernel<32>{}));
                }
                else if(warpExtent == 64)
                {
                    REQUIRE(fixture(AnyMultipleThreadWarpTestKernel<64>{}));
                }
            }
        }
    }
}

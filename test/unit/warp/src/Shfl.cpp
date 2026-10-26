/* Copyright 2023 David M. Rogers, Jan Stephan, Andrea Bocci, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#include "SyclWarpSupport.hpp"

#include <aV2P/math/FloatEqualExact.hpp>
#include <aV2P/test/KernelExecutionFixture.hpp>
#include <aV2P/test/acc/TestAccs.hpp>
#include <aV2P/test/queue/Queue.hpp>
#include <aV2P/warp/Traits.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <limits>

struct ShflSingleThreadWarpTestKernel
{
    AV2P_NO_HOST_ACC_WARNING
    template<typename TAcc>
    AV2P_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        if constexpr(aV2P::Dim<TAcc>::value > 0)
        {
            AV2P_CHECK(*success, aV2P::warp::getSize(acc) == 1);
            AV2P_CHECK(*success, aV2P::warp::shfl(acc, 42, -1) == 42);
        }
        else
        {
            AV2P_CHECK(*success, aV2P::warp::shfl(acc, 42, 0, 1) == 42);
        }
        AV2P_CHECK(*success, aV2P::warp::shfl(acc, 12, 0) == 12);
        float ans = aV2P::warp::shfl(acc, 3.3f, 0);
        AV2P_CHECK(*success, aV2P::math::floatEqualExactNoWarning(ans, 3.3f));
    }
};

template<std::uint32_t TWarpSize>
struct ShflMultipleThreadWarpTestKernel
{
    AV2P_NO_HOST_ACC_WARNING
    template<typename TAcc>
    AV2P_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        auto const localThreadIdx = aV2P::getIdx<aV2P::Block, aV2P::Threads>(acc);
        auto const blockExtent = aV2P::getWorkDiv<aV2P::Block, aV2P::Threads>(acc);
        std::int32_t const warpExtent = aV2P::warp::getSize(acc);
        // Test relies on having a single warp per thread block
        AV2P_CHECK(*success, static_cast<std::int32_t>(blockExtent.prod()) == warpExtent);
        auto const threadIdxInWarp = std::int32_t(aV2P::mapIdx<1u>(localThreadIdx, blockExtent)[0]);

        AV2P_CHECK(*success, warpExtent > 1);

        AV2P_CHECK(*success, aV2P::warp::shfl(acc, 42, 0) == 42);
        AV2P_CHECK(*success, aV2P::warp::shfl(acc, threadIdxInWarp, 0) == 0);
        AV2P_CHECK(*success, aV2P::warp::shfl(acc, threadIdxInWarp, 1) == 1);
        // Note the CUDA and HIP API-s differ on lane wrapping, but both agree it should not segfault
        // https://github.com/ROCm-Developer-Tools/HIP-CPU/issues/14
        AV2P_CHECK(*success, aV2P::warp::shfl(acc, 5, -1) == 5);

        auto const epsilon = std::numeric_limits<float>::epsilon();

        // Test various widths
        for(int width = 1; width < warpExtent; width *= 2)
        {
            for(int idx = 0; idx < width; idx++)
            {
                int const off = width * (threadIdxInWarp / width);
                AV2P_CHECK(*success, aV2P::warp::shfl(acc, threadIdxInWarp, idx, width) == idx + off);
                float const ans = aV2P::warp::shfl(acc, 4.0f - float(threadIdxInWarp), idx, width);
                float const expect = 4.0f - float(idx + off);
                AV2P_CHECK(*success, aV2P::math::abs(acc, ans - expect) < epsilon);
            }
        }

        // Some threads quit the kernel to test that the warp operations
        // properly operate on the active threads only
        if(threadIdxInWarp >= warpExtent / 2)
            return;

        for(int idx = 0; idx < warpExtent / 2; idx++)
        {
            AV2P_CHECK(*success, aV2P::warp::shfl(acc, threadIdxInWarp, idx) == idx);
            float const ans = aV2P::warp::shfl(acc, 4.0f - float(threadIdxInWarp), idx);
            float const expect = 4.0f - float(idx);
            AV2P_CHECK(*success, aV2P::math::abs(acc, ans - expect) < epsilon);
        }
    }
};

template<std::uint32_t TWarpSize, typename TAcc>
struct aV2P::trait::WarpSize<ShflMultipleThreadWarpTestKernel<TWarpSize>, TAcc>
    : std::integral_constant<std::uint32_t, TWarpSize>
{
};

TEMPLATE_LIST_TEST_CASE("shfl", "[warp]", aV2P::test::TestAccs)
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
                REQUIRE(fixture(ShflSingleThreadWarpTestKernel{}));
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
                    REQUIRE(fixture(ShflMultipleThreadWarpTestKernel<4>{}));
                }
                else if(warpExtent == 8)
                {
                    REQUIRE(fixture(ShflMultipleThreadWarpTestKernel<8>{}));
                }
                else if(warpExtent == 16)
                {
                    REQUIRE(fixture(ShflMultipleThreadWarpTestKernel<16>{}));
                }
                else if(warpExtent == 32)
                {
                    REQUIRE(fixture(ShflMultipleThreadWarpTestKernel<32>{}));
                }
                else if(warpExtent == 64)
                {
                    REQUIRE(fixture(ShflMultipleThreadWarpTestKernel<64>{}));
                }
            }
        }
    }
}

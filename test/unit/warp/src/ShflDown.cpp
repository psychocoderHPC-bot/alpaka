/* Copyright 2023 Aurora Perego
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
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

#if AV2P_COMP_GNUC
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wstrict-overflow"
#endif

struct ShflDownSingleThreadWarpTestKernel
{
    AV2P_NO_HOST_ACC_WARNING
    template<typename TAcc>
    AV2P_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        if constexpr(aV2P::Dim<TAcc>::value > 0)
        {
            AV2P_CHECK(*success, aV2P::warp::getSize(acc) == 1);
            AV2P_CHECK(*success, aV2P::warp::shfl_down(acc, 42, 0) == 42);
        }
        else
        {
            AV2P_CHECK(*success, aV2P::warp::shfl_down(acc, 42, 0, 1) == 42);
        }
        AV2P_CHECK(*success, aV2P::warp::shfl_down(acc, 12, 0) == 12);
        float ans = aV2P::warp::shfl_down(acc, 3.3f, 0);
        AV2P_CHECK(*success, aV2P::math::floatEqualExactNoWarning(ans, 3.3f));
    }
};

template<std::uint32_t TWarpSize>
struct ShflDownMultipleThreadWarpTestKernel
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

        AV2P_CHECK(*success, aV2P::warp::shfl_down(acc, 42, 0) == 42);
        AV2P_CHECK(*success, aV2P::warp::shfl_down(acc, threadIdxInWarp, 0) == threadIdxInWarp);
        AV2P_CHECK(
            *success,
            aV2P::warp::shfl_down(acc, threadIdxInWarp, 1)
                == (threadIdxInWarp + 1 < warpExtent ? threadIdxInWarp + 1 : threadIdxInWarp));
        auto const epsilon = std::numeric_limits<float>::epsilon();

        // Test various widths
        for(int width = 1; width < warpExtent; width *= 2)
        {
            for(int idx = 0; idx < width; idx++)
            {
                int const off = width * (threadIdxInWarp / width);
                AV2P_CHECK(
                    *success,
                    aV2P::warp::shfl_down(acc, threadIdxInWarp, static_cast<std::uint32_t>(idx), width)
                        == ((threadIdxInWarp + idx < (width + off)) ? threadIdxInWarp + idx : threadIdxInWarp));
                float const ans = aV2P::warp::shfl_down(
                    acc,
                    4.0f - float(threadIdxInWarp),
                    static_cast<std::uint32_t>(idx),
                    width);
                float const expect
                    = ((threadIdxInWarp + idx < (width + off)) ? (4.0f - float(threadIdxInWarp + idx))
                                                               : (4.0f - float(threadIdxInWarp)));
                AV2P_CHECK(*success, aV2P::math::abs(acc, ans - expect) < epsilon);
            }
        }

        // Some threads quit the kernel to test that the warp operations
        // properly operate on the active threads only
        if(threadIdxInWarp >= warpExtent / 2)
            return;

        for(int idx = 0; idx < warpExtent / 2; idx++)
        {
            auto const shfl = aV2P::warp::shfl_down(acc, threadIdxInWarp, static_cast<std::uint32_t>(idx));
            float const ans
                = aV2P::warp::shfl_down(acc, 4.0f - float(threadIdxInWarp), static_cast<std::uint32_t>(idx));
            float const expect
                = ((threadIdxInWarp + idx < warpExtent / 2) ? (4.0f - float(threadIdxInWarp + idx)) : 0);
            if(threadIdxInWarp + idx < warpExtent / 2)
            {
                AV2P_CHECK(*success, shfl == threadIdxInWarp + idx);
                AV2P_CHECK(*success, aV2P::math::abs(acc, ans - expect) < epsilon);
            }
        }
    }
};

template<std::uint32_t TWarpSize, typename TAcc>
struct aV2P::trait::WarpSize<ShflDownMultipleThreadWarpTestKernel<TWarpSize>, TAcc>
    : std::integral_constant<std::uint32_t, TWarpSize>
{
};

TEMPLATE_LIST_TEST_CASE("shfl_down", "[warp]", aV2P::test::TestAccs)
{
    using Acc = TestType;

    if constexpr(constexpr char const* reason = aV2P::test::warpUnsupportedReason<Acc>(); reason != nullptr)
    {
        WARN("Test disabled: " << reason);
    }
    else
    {
        using Dev = aV2P::Dev<Acc>;
        using Dim = aV2P::Dim<Acc>;
        using Idx = aV2P::Idx<Acc>;

        auto const platform = aV2P::Platform<Acc>{};
        Dev const dev(aV2P::getDevByIdx(platform, 0u));
        auto const warpExtents = aV2P::getWarpSizes(dev);
        for(auto const warpExtent : warpExtents)
        {
            auto const scalar = Dim::value == 0 || warpExtent == 1;
            if(scalar)
            {
                aV2P::test::KernelExecutionFixture<Acc> fixture(aV2P::Vec<Dim, Idx>::all(4));
                REQUIRE(fixture(ShflDownSingleThreadWarpTestKernel{}));
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
                    REQUIRE(fixture(ShflDownMultipleThreadWarpTestKernel<4>{}));
                }
                else if(warpExtent == 8)
                {
                    REQUIRE(fixture(ShflDownMultipleThreadWarpTestKernel<8>{}));
                }
                else if(warpExtent == 16)
                {
                    REQUIRE(fixture(ShflDownMultipleThreadWarpTestKernel<16>{}));
                }
                else if(warpExtent == 32)
                {
                    REQUIRE(fixture(ShflDownMultipleThreadWarpTestKernel<32>{}));
                }
                else if(warpExtent == 64)
                {
                    REQUIRE(fixture(ShflDownMultipleThreadWarpTestKernel<64>{}));
                }
            }
        }
    }
}

#if AV2P_COMP_GNUC
#    pragma GCC diagnostic pop
#endif

/* Copyright 2026 Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/test/KernelExecutionFixture.hpp>
#include <aV2P/warp/Traits.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <stdexcept>

#if defined(AV2P_ACC_SYCL_ENABLED) && defined(AV2P_SYCL_ONEAPI_CPU)

#    include <sycl/sycl.hpp>

template<std::uint32_t TWarpSize>
struct SyclCpuWarpLayoutTestKernel
{
    template<typename TAcc>
    AV2P_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        AV2P_CHECK(*success, aV2P::warp::getSize(acc) == static_cast<std::int32_t>(TWarpSize));
    }
};

template<std::uint32_t TWarpSize, typename TAcc>
struct aV2P::trait::WarpSize<SyclCpuWarpLayoutTestKernel<TWarpSize>, TAcc>
    : std::integral_constant<std::uint32_t, TWarpSize>
{
};

// On the SYCL CPU device sub-groups are formed only along the innermost dimension of the work-group, so a kernel that
// requires a warp size must be launched with an innermost block extent that is a multiple of the warp size.
template<std::uint32_t TWarpSize>
void testSyclCpuWarpLayout()
{
    using Dim = aV2P::DimInt<2u>;
    using Idx = std::uint32_t;
    using Acc = aV2P::AccCpuSycl<Dim, Idx>;
    using Vec = aV2P::Vec<Dim, Idx>;
    using ExecutionFixture = aV2P::test::KernelExecutionFixture<Acc>;

    // the warp is laid out along the innermost dimension: valid
    auto good = ExecutionFixture{typename ExecutionFixture::WorkDiv{Vec{2, 2}, Vec{1, TWarpSize}, Vec{1, 1}}};
    CHECK(good(SyclCpuWarpLayoutTestKernel<TWarpSize>{}));

    // the warp is laid out along the outer dimension: rejected when the kernel is launched
    auto bad = ExecutionFixture{typename ExecutionFixture::WorkDiv{Vec{2, 2}, Vec{TWarpSize, 1}, Vec{1, 1}}};
    CHECK_THROWS_AS(bad(SyclCpuWarpLayoutTestKernel<TWarpSize>{}), std::runtime_error);
}

TEST_CASE("syclCpuWarpLayout", "[warp]")
{
    using Acc = aV2P::AccCpuSycl<aV2P::DimInt<2u>, std::uint32_t>;
    auto const platform = aV2P::Platform<Acc>{};
    auto const dev = aV2P::getDevByIdx(platform, 0);
    for(auto const warpSize : aV2P::getWarpSizes(dev))
    {
        if(warpSize == 4)
            testSyclCpuWarpLayout<4>();
        else if(warpSize == 8)
            testSyclCpuWarpLayout<8>();
        else if(warpSize == 16)
            testSyclCpuWarpLayout<16>();
        else if(warpSize == 32)
            testSyclCpuWarpLayout<32>();
    }
}

#endif

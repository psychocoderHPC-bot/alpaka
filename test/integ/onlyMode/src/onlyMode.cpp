/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

// Regression test for AV2P_ACC_GPU_CUDA_ONLY_MODE / AV2P_ACC_GPU_HIP_ONLY_MODE.
//
// ONLY_MODE used to define AV2P_FN_ACC as `__device__`, which made the kernel functor
// non-invocable on the host.  Native alpaka 3 checks host invocability before it launches a
// kernel, so the functor was rejected and every kernel launch failed.  This test only passes
// when AV2P_FN_ACC is `__device__ __host__` in only mode.

#include <aV2P/alpaka.hpp>
#include <aV2P/test/KernelExecutionFixture.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <type_traits>

#if defined(AV2P_ACC_GPU_CUDA_ONLY_MODE) && defined(AV2P_ACC_GPU_CUDA_ENABLED) && AV2P_LANG_CUDA
#    define AV2P_ONLY_MODE_TEST 1
#endif
#if defined(AV2P_ACC_GPU_HIP_ONLY_MODE) && defined(AV2P_ACC_GPU_HIP_ENABLED) && AV2P_LANG_HIP
#    define AV2P_ONLY_MODE_TEST 1
#endif

#if defined(AV2P_ONLY_MODE_TEST)

struct OnlyModeKernel
{
    template<typename TAcc>
    AV2P_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        // The functor must be invocable on the host and on the device.
        if constexpr(!std::is_same_v<TAcc, void>)
            *success = true;
    }
};

TEST_CASE("onlyModeKernelLaunch", "[onlyMode]")
{
#    if defined(AV2P_ACC_GPU_CUDA_ONLY_MODE)
    using TAcc = aV2P::AccGpuCudaRt<aV2P::DimInt<1u>, std::uint32_t>;
#    else
    using TAcc = aV2P::AccGpuHipRt<aV2P::DimInt<1u>, std::uint32_t>;
#    endif
    using Dim = aV2P::Dim<TAcc>;
    using Idx = aV2P::Idx<TAcc>;

    aV2P::test::KernelExecutionFixture<TAcc> fixture(aV2P::Vec<Dim, Idx>::ones());

    REQUIRE(fixture(OnlyModeKernel{}));
}

#endif

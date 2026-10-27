/* Copyright 2022 Axel Huebl, Benjamin Worpitz, René Widera, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/alpaka.hpp>
#include <aV2P/test/KernelExecutionFixture.hpp>
#include <catch2/catch_test_macros.hpp>

#if defined(AV2P_ACC_GPU_CUDA_ONLY_MODE) && defined(AV2P_ACC_GPU_CUDA_ENABLED) && AV2P_LANG_CUDA

//! Native CUDA function.
#    if AV2P_COMP_CLANG
#        pragma clang diagnostic push
#        pragma clang diagnostic ignored "-Wmissing-prototypes"
#    endif
__device__ static auto userDefinedThreadFence() -> void
{
    __threadfence();
}
#    if AV2P_COMP_CLANG
#        pragma clang diagnostic pop
#    endif

class CudaOnlyTestKernel
{
public:
    template<typename TAcc>
    AV2P_FN_ACC auto operator()(TAcc const& /* acc */, bool* success) const -> void
    {
        // We should be able to call some native CUDA functions when AV2P_ACC_GPU_CUDA_ONLY_MODE is enabled.
        __threadfence_block();
        userDefinedThreadFence();
        __threadfence_system();

        *success = true;
    }
};

TEST_CASE("cudaOnlyModeWorking", "[cudaOnly]")
{
    using TAcc = aV2P::AccGpuCudaRt<aV2P::DimInt<1u>, std::uint32_t>;
    using Dim = aV2P::Dim<TAcc>;
    using Idx = aV2P::Idx<TAcc>;

    aV2P::test::KernelExecutionFixture<TAcc> fixture(aV2P::Vec<Dim, Idx>::ones());

    CudaOnlyTestKernel kernel;

    REQUIRE(fixture(kernel));
}

#endif

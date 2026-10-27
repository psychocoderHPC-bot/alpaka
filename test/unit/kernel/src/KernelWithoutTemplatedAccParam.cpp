/* Copyright 2024 Axel Huebl, Benjamin Worpitz, Matthias Werner, Bernhard Manfred Gruber, Jan Stephan, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/kernel/Traits.hpp>
#include <aV2P/test/KernelExecutionFixture.hpp>
#include <catch2/catch_test_macros.hpp>

using Dim = aV2P::DimInt<2u>;
using Idx = std::uint32_t;
#if defined(AV2P_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
using AccCpu = aV2P::AccCpuSerial<Dim, Idx>;
#endif
#if defined(AV2P_ACC_GPU_HIP_ENABLED) && AV2P_LANG_HIP
using AccGpu = aV2P::AccGpuHipRt<Dim, Idx>;
#elif defined(AV2P_ACC_GPU_CUDA_ENABLED) && AV2P_LANG_CUDA
using AccGpu = aV2P::AccGpuCudaRt<Dim, Idx>;
#endif

#if defined(AV2P_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
struct KernelNoTemplateCpu
{
    AV2P_FN_ACC auto operator()(AccCpu const& acc, bool* success) const -> void
    {
        AV2P_CHECK(
            *success,
            static_cast<aV2P::Idx<AccCpu>>(1) == (aV2P::getWorkDiv<aV2P::Grid, aV2P::Threads>(acc)).prod());
    }
};

TEST_CASE("kernelNoTemplateCpu", "[kernel]")
{
    aV2P::test::KernelExecutionFixture<AccCpu> fixture(aV2P::Vec<Dim, Idx>::ones());

    KernelNoTemplateCpu kernel;

    REQUIRE(fixture(kernel));
}
#endif

#if(defined(AV2P_ACC_GPU_CUDA_ENABLED) && AV2P_LANG_CUDA) || (defined(AV2P_ACC_GPU_HIP_ENABLED) && AV2P_LANG_HIP)
struct KernelNoTemplateGpu
{
    AV2P_FN_ACC
    auto operator()(AccGpu const& acc, bool* success) const -> void
    {
        AV2P_CHECK(
            *success,
            static_cast<aV2P::Idx<AccGpu>>(1) == (aV2P::getWorkDiv<aV2P::Grid, aV2P::Threads>(acc)).prod());
    }
};

TEST_CASE("kernelNoTemplateGpu", "[kernel]")
{
    aV2P::test::KernelExecutionFixture<AccGpu> fixture(aV2P::Vec<Dim, Idx>::ones());

    KernelNoTemplateGpu kernel;

    REQUIRE(fixture(kernel));
}
#endif

#if defined(AV2P_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
struct KernelUnusedTemplateParamCpu
{
    template<typename TNotUsed = void>
    AV2P_FN_ACC auto operator()(AccCpu const& acc, bool* success) const -> void
    {
        AV2P_CHECK(
            *success,
            static_cast<aV2P::Idx<AccCpu>>(1) == (aV2P::getWorkDiv<aV2P::Grid, aV2P::Threads>(acc)).prod());
    }
};

TEST_CASE("kernelUnusedTemplateParamCpu", "[kernel]")
{
    aV2P::test::KernelExecutionFixture<AccCpu> fixture(aV2P::Vec<Dim, Idx>::ones());

    KernelUnusedTemplateParamCpu kernel;

    REQUIRE(fixture(kernel));
}
#endif

#if(defined(AV2P_ACC_GPU_CUDA_ENABLED) && AV2P_LANG_CUDA) || (defined(AV2P_ACC_GPU_HIP_ENABLED) && AV2P_LANG_HIP)
struct KernelUnusedTemplateParamGpu
{
    template<typename TNotUsed = void>
    AV2P_FN_ACC auto operator()(AccGpu const& acc, bool* success) const -> void
    {
        AV2P_CHECK(
            *success,
            static_cast<aV2P::Idx<AccGpu>>(1) == (aV2P::getWorkDiv<aV2P::Grid, aV2P::Threads>(acc)).prod());
    }
};

TEST_CASE("kernelUnusedTemplateParamGpu", "[kernel]")
{
    aV2P::test::KernelExecutionFixture<AccGpu> fixture(aV2P::Vec<Dim, Idx>::ones());

    KernelUnusedTemplateParamGpu kernel;

    REQUIRE(fixture(kernel));
}
#endif

/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

// A captureless device lambda is a valid aV2P kernel functor.  Its closure type is empty and can be copied to the
// device even on implementations that do not classify the closure type as trivially copyable.  Capturing lambdas
// remain unsupported and must be converted to a named functor.
#include <aV2P/kernel/Traits.hpp>
#include <aV2P/test/KernelExecutionFixture.hpp>
#include <aV2P/test/acc/TestAccs.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <type_traits>

namespace
{
    //! A non-trivially-copyable capture used to model a capturing closure in the negative compile-time check.
    struct NonTrivialCapture
    {
        NonTrivialCapture() = default;

        NonTrivialCapture(NonTrivialCapture const&)
        {
        }
    };
} // namespace

// A capturing lambda is not a valid kernel functor.  The plain (host) lambda is not an extended device lambda, so
// this check is independent of the CUDA extended-lambda trait.
static_assert(!aV2P::isKernelTriviallyCopyable<decltype([capture = NonTrivialCapture{}]() {})>);

#if defined(AV2P_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
using CpuDim = aV2P::DimInt<1u>;
using CpuIdx = std::uint32_t;
using CpuAcc = aV2P::AccCpuSerial<CpuDim, CpuIdx>;

TEST_CASE("capturelessLambdaKernelCpu", "[kernel]")
{
    aV2P::test::KernelExecutionFixture<CpuAcc> fixture(aV2P::Vec<CpuDim, CpuIdx>::ones());
    REQUIRE(fixture(
        [] AV2P_FN_ACC(CpuAcc const& acc, bool* success)
        {
            AV2P_CHECK(*success, (aV2P::getWorkDiv<aV2P::Grid, aV2P::Threads>(acc).prod() == static_cast<CpuIdx>(1)));
            // Use device primitives through the concrete accelerator facade.
            auto& value = aV2P::declareSharedVar<int, 0>(acc);
            value = 1;
            AV2P_CHECK(*success, (aV2P::atomicOp<aV2P::AtomicAdd>(acc, &value, 1) == 1));
            AV2P_CHECK(*success, value == 2);
        }));
}
#endif

#if defined(AV2P_ACC_GPU_CUDA_ENABLED) && AV2P_LANG_CUDA
using CudaDim = aV2P::DimInt<1u>;
using CudaIdx = std::uint32_t;
using CudaAcc = aV2P::AccGpuCudaRt<CudaDim, CudaIdx>;

TEST_CASE("capturelessLambdaKernelCuda", "[kernel]")
{
    aV2P::test::KernelExecutionFixture<CudaAcc> fixture(aV2P::Vec<CudaDim, CudaIdx>::ones());
    REQUIRE(fixture(
        [] AV2P_FN_ACC(CudaAcc const& acc, bool* success) {
            AV2P_CHECK(*success, (aV2P::getWorkDiv<aV2P::Grid, aV2P::Threads>(acc).prod() == static_cast<CudaIdx>(1)));
        }));
}
#endif

#if defined(AV2P_ACC_GPU_HIP_ENABLED) && AV2P_LANG_HIP
using HipDim = aV2P::DimInt<1u>;
using HipIdx = std::uint32_t;
using HipAcc = aV2P::AccGpuHipRt<HipDim, HipIdx>;

TEST_CASE("capturelessLambdaKernelHip", "[kernel]")
{
    aV2P::test::KernelExecutionFixture<HipAcc> fixture(aV2P::Vec<HipDim, HipIdx>::ones());
    REQUIRE(fixture(
        [] AV2P_FN_ACC(HipAcc const& acc, bool* success) {
            AV2P_CHECK(*success, (aV2P::getWorkDiv<aV2P::Grid, aV2P::Threads>(acc).prod() == static_cast<HipIdx>(1)));
        }));
}
#endif

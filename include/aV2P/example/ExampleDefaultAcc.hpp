/* Copyright 2025 Jeffrey Kelling, Bernhard Manfred Gruber, Jan Stephan, Aurora Perego, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include "aV2P/alpaka.hpp"

#pragma once

namespace aV2P
{
    //! Alias for the default accelerator used by examples. From a list of
    //! all accelerators the first one which is enabled is chosen.
    //! AccCpuSerial is selected last.
    template<class TDim, class TIdx>
#if defined(AV2P_ACC_GPU_CUDA_ENABLED)
    using ExampleDefaultAcc = aV2P::AccGpuCudaRt<TDim, TIdx>;
#elif defined(AV2P_ACC_GPU_HIP_ENABLED)
    using ExampleDefaultAcc = aV2P::AccGpuHipRt<TDim, TIdx>;
#elif defined(AV2P_ACC_CPU_B_OMP2_T_SEQ_ENABLED)
    using ExampleDefaultAcc = aV2P::AccCpuOmp2Blocks<TDim, TIdx>;
#elif defined(AV2P_ACC_CPU_B_TBB_T_SEQ_ENABLED)
    using ExampleDefaultAcc = aV2P::AccCpuTbbBlocks<TDim, TIdx>;
#elif defined(AV2P_ACC_CPU_B_SEQ_T_OMP2_ENABLED)
    using ExampleDefaultAcc = aV2P::AccCpuOmp2Threads<TDim, TIdx>;
#elif defined(AV2P_ACC_CPU_B_SEQ_T_THREADS_ENABLED)
    using ExampleDefaultAcc = aV2P::AccCpuThreads<TDim, TIdx>;
#elif defined(AV2P_ACC_SYCL_ENABLED)
// AV2P_SYCL_TARGET_* are the device-kind macros on the native alpaka 3 path, AV2P_SYCL_ONEAPI_* the
// ones on the self-contained backward-compatible path; both name the same selection.
#    if defined(AV2P_SYCL_ONEAPI_CPU) || defined(AV2P_SYCL_TARGET_CPU)
    using ExampleDefaultAcc = aV2P::AccCpuSycl<TDim, TIdx>;
#    elif defined(AV2P_SYCL_ONEAPI_GPU) || defined(AV2P_SYCL_TARGET_GPU)
    using ExampleDefaultAcc = aV2P::AccGpuSyclIntel<TDim, TIdx>;
#    elif defined(AV2P_SYCL_ONEAPI_GPU_NVIDIA) || defined(AV2P_SYCL_TARGET_GPU_NVIDIA)
    using ExampleDefaultAcc = aV2P::AccGpuSyclNvidia<TDim, TIdx>;
#    elif defined(AV2P_SYCL_ONEAPI_GPU_AMD) || defined(AV2P_SYCL_TARGET_GPU_AMD)
    using ExampleDefaultAcc = aV2P::AccGpuSyclAmd<TDim, TIdx>;
#    endif
#elif defined(AV2P_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
    using ExampleDefaultAcc = aV2P::AccCpuSerial<TDim, TIdx>;
#else
    class ExampleDefaultAcc;
#    warning "No supported backend selected."
#endif
} // namespace aV2P

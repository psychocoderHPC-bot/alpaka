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
#    if defined(AV2P_SYCL_ONEAPI_CPU)
    using ExampleDefaultAcc = aV2P::AccCpuSycl<TDim, TIdx>;
#    elif defined(AV2P_SYCL_ONEAPI_FPGA)
    using ExampleDefaultAcc = aV2P::AccFpgaSyclIntel<TDim, TIdx>;
#    elif defined(AV2P_SYCL_ONEAPI_GPU)
    using ExampleDefaultAcc = aV2P::AccGpuSyclIntel<TDim, TIdx>;
#    elif defined(AV2P_SYCL_ONEAPI_GPU_NVIDIA)
    using ExampleDefaultAcc = aV2P::AccGpuSyclNvidia<TDim, TIdx>;
#    elif defined(AV2P_SYCL_ONEAPI_GPU_AMD)
    using ExampleDefaultAcc = aV2P::AccGpuSyclAmd<TDim, TIdx>;
#    endif
#elif defined(AV2P_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
    using ExampleDefaultAcc = aV2P::AccCpuSerial<TDim, TIdx>;
#else
    class ExampleDefaultAcc;
#    warning "No supported backend selected."
#endif
} // namespace aV2P

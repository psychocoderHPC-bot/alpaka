/* Copyright 2022 Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/queue/cuda-hip/QueueUniformCudaHipRt.hpp"

#if defined(AV2P_ACC_GPU_CUDA_ENABLED) || defined(AV2P_ACC_GPU_HIP_ENABLED)

namespace aV2P
{
    //! The CUDA/HIP RT blocking queue.
    template<typename TApi>
    using QueueUniformCudaHipRtBlocking = uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, true>;

} // namespace aV2P

#endif

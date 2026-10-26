/* Copyright 2026 Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/ApiHipRt.hpp"
#include "aV2P/kernel/TaskKernelGpuUniformCudaHipRt.hpp"

#ifdef AV2P_ACC_GPU_HIP_ENABLED

namespace aV2P
{
    template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, bool TCooperative, typename... TArgs>
    using TaskKernelGpuCudaHipRt
        = TaskKernelGpuUniformCudaHipRt<ApiHipRt, TAcc, TDim, TIdx, TKernelFnObj, TCooperative, TArgs...>;

} // namespace aV2P

#endif // AV2P_ACC_GPU_HIP_ENABLED

/* Copyright 2026 Aurora Perego, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/acc/Tag.hpp"
#include "aV2P/kernel/TaskKernelGenericSycl.hpp"

#if defined(AV2P_ACC_SYCL_ENABLED) && defined(AV2P_SYCL_ONEAPI_GPU_AMD)

namespace aV2P
{
    template<typename TDim, typename TIdx, typename TKernelFnObj, bool TCooperative, typename... TArgs>
    using TaskKernelGpuSyclAmd = TaskKernelGenericSycl<
        TagGpuSyclAmd,
        AccGpuSyclAmd<TDim, TIdx>,
        TDim,
        TIdx,
        TKernelFnObj,
        TCooperative,
        TArgs...>;

} // namespace aV2P

#endif

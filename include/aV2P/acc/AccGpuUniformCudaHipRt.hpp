/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

//! Legacy accelerator name kept for source compatibility; the implementation is the alpaka 3 adapter.
//!
//! The first (now unused) template parameter is the legacy API tag; it is kept so that existing
//! code can spell the three-parameter legacy name.  The second API tag path maps to the corresponding
//! accelerator of the enabled GPU backend.

#include "aV2P/adapter/Acc.hpp"

namespace aV2P
{
#if defined(AV2P_ACC_GPU_CUDA_ENABLED)
    template<typename TApi, typename TDim, typename TIdx>
    using AccGpuUniformCudaHipRt = AccGpuCudaRt<TDim, TIdx>;
#elif defined(AV2P_ACC_GPU_HIP_ENABLED)
    template<typename TApi, typename TDim, typename TIdx>
    using AccGpuUniformCudaHipRt = AccGpuHipRt<TDim, TIdx>;
#endif
} // namespace aV2P

/* Copyright 2022 Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/acc/AccGpuUniformCudaHipRt.hpp"
#include "aV2P/acc/Tag.hpp"
#include "aV2P/core/ApiCudaRt.hpp"

#ifdef AV2P_ACC_GPU_CUDA_ENABLED

namespace aV2P
{
    template<typename TDim, typename TIdx>
    using AccGpuCudaRt = AccGpuUniformCudaHipRt<ApiCudaRt, TDim, TIdx>;

    namespace trait
    {
        template<typename TDim, typename TIdx>
        struct AccToTag<aV2P::AccGpuCudaRt<TDim, TIdx>>
        {
            using type = aV2P::TagGpuCudaRt;
        };

        template<typename TDim, typename TIdx>
        struct TagToAcc<aV2P::TagGpuCudaRt, TDim, TIdx>
        {
            using type = aV2P::AccGpuCudaRt<TDim, TIdx>;
        };
    } // namespace trait
} // namespace aV2P

#endif // AV2P_ACC_GPU_CUDA_ENABLED

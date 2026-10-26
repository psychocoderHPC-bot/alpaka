/* Copyright 2022 Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/acc/AccGpuUniformCudaHipRt.hpp"
#include "aV2P/acc/Tag.hpp"
#include "aV2P/core/ApiHipRt.hpp"

#ifdef AV2P_ACC_GPU_HIP_ENABLED

namespace aV2P
{
    template<typename TDim, typename TIdx>
    using AccGpuHipRt = AccGpuUniformCudaHipRt<ApiHipRt, TDim, TIdx>;

    namespace trait
    {
        template<typename TDim, typename TIdx>
        struct AccToTag<aV2P::AccGpuHipRt<TDim, TIdx>>
        {
            using type = aV2P::TagGpuHipRt;
        };

        template<typename TDim, typename TIdx>
        struct TagToAcc<aV2P::TagGpuHipRt, TDim, TIdx>
        {
            using type = aV2P::AccGpuHipRt<TDim, TIdx>;
        };
    } // namespace trait
} // namespace aV2P

#endif // AV2P_ACC_GPU_HIP_ENABLED

/* Copyright 2022 Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/ApiCudaRt.hpp"
#include "aV2P/mem/buf/uniformCudaHip/BufUniformCudaHipRt.hpp"

#ifdef AV2P_ACC_GPU_CUDA_ENABLED

namespace aV2P
{
    template<typename TElem, typename TDim, typename TIdx>
    using ConstBufCudaRt = ConstBufUniformCudaHipRt<ApiCudaRt, TElem, TDim, TIdx>;

    template<typename TElem, typename TDim, typename TIdx>
    using BufCudaRt = BufUniformCudaHipRt<ApiCudaRt, TElem, TDim, TIdx>;
} // namespace aV2P

#endif // AV2P_ACC_GPU_CUDA_ENABLED

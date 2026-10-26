/* Copyright 2022 Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/ApiCudaRt.hpp"
#include "aV2P/dev/DevUniformCudaHipRt.hpp"

#ifdef AV2P_ACC_GPU_CUDA_ENABLED

namespace aV2P
{
    //! The CUDA RT device handle.
    using DevCudaRt = DevUniformCudaHipRt<ApiCudaRt>;
} // namespace aV2P

#endif // AV2P_ACC_GPU_CUDA_ENABLED

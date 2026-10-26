/* Copyright 2022 Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/ApiCudaRt.hpp"
#include "aV2P/platform/PlatformUniformCudaHipRt.hpp"

#ifdef AV2P_ACC_GPU_CUDA_ENABLED

namespace aV2P
{
    //! The CUDA RT platform.
    using PlatformCudaRt = PlatformUniformCudaHipRt<ApiCudaRt>;
} // namespace aV2P

#endif // AV2P_ACC_GPU_CUDA_ENABLED

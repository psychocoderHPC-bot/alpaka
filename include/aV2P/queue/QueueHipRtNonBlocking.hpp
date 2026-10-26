/* Copyright 2022 Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/ApiHipRt.hpp"
#include "aV2P/queue/QueueUniformCudaHipRtNonBlocking.hpp"

#ifdef AV2P_ACC_GPU_HIP_ENABLED

namespace aV2P
{
    //! The HIP RT non-blocking queue.
    using QueueHipRtNonBlocking = QueueUniformCudaHipRtNonBlocking<ApiHipRt>;
} // namespace aV2P

#endif // AV2P_ACC_GPU_HIP_ENABLED

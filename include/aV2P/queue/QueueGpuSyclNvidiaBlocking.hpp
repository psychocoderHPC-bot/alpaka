/* Copyright 2025 Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/acc/Tag.hpp"
#include "aV2P/queue/QueueGenericSyclBlocking.hpp"

#if defined(AV2P_ACC_SYCL_ENABLED) && defined(AV2P_SYCL_ONEAPI_GPU_NVIDIA)

namespace aV2P
{
    using QueueGpuSyclNvidiaBlocking = QueueGenericSyclBlocking<TagGpuSyclNvidia>;
} // namespace aV2P

#endif

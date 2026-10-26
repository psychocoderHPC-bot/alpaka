/* Copyright 2024 Jan Stephan, Luca Ferragina, Andrea Bocci, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/acc/Tag.hpp"
#include "aV2P/queue/QueueGenericSyclBlocking.hpp"

#if defined(AV2P_ACC_SYCL_ENABLED) && defined(AV2P_SYCL_ONEAPI_CPU)

namespace aV2P
{
    using QueueCpuSyclBlocking = QueueGenericSyclBlocking<TagCpuSycl>;
} // namespace aV2P

#endif

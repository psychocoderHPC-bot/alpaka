/* Copyright 2024 Jan Stephan, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/queue/sycl/QueueGenericSyclBase.hpp"

#ifdef AV2P_ACC_SYCL_ENABLED

namespace aV2P
{
    template<concepts::Tag TTag>
    using QueueGenericSyclNonBlocking = detail::QueueGenericSyclBase<TTag, false>;
} // namespace aV2P

#endif

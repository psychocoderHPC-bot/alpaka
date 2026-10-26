/* Copyright 2024 Jan Stephan, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/acc/Tag.hpp"
#include "aV2P/dev/DevGenericSycl.hpp"

#if defined(AV2P_ACC_SYCL_ENABLED) && defined(AV2P_SYCL_ONEAPI_FPGA)

namespace aV2P
{
    using DevFpgaSyclIntel = DevGenericSycl<TagFpgaSyclIntel>;
} // namespace aV2P

#endif

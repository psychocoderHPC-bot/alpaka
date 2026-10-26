/* Copyright 2024 Jan Stephan, Luca Ferragina, Andrea Bocci, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/acc/Tag.hpp"
#include "aV2P/dev/DevGenericSycl.hpp"
#include "aV2P/dev/Traits.hpp"
#include "aV2P/platform/PlatformGenericSycl.hpp"

#if defined(AV2P_ACC_SYCL_ENABLED) && defined(AV2P_SYCL_ONEAPI_CPU)

#    include <sycl/sycl.hpp>

namespace aV2P
{
    namespace detail
    {
        template<>
        struct SYCLDeviceSelector<TagCpuSycl>
        {
            auto operator()(sycl::device const& dev) const -> int
            {
                return dev.is_cpu() ? 1 : -1;
            }
        };
    } // namespace detail

    //! The SYCL device manager.
    using PlatformCpuSycl = PlatformGenericSycl<TagCpuSycl>;
} // namespace aV2P

#endif

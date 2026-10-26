/* Copyright 2025 Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/dev/DevGenericSycl.hpp"
#include "aV2P/dev/Traits.hpp"
#include "aV2P/platform/PlatformGenericSycl.hpp"

#if defined(AV2P_ACC_SYCL_ENABLED) && defined(AV2P_SYCL_ONEAPI_GPU_AMD)

#    include <sycl/sycl.hpp>

namespace aV2P
{
    namespace detail
    {
        template<>
        struct SYCLDeviceSelector<TagGpuSyclAmd>
        {
            auto operator()(sycl::device const& dev) const -> int
            {
                auto const& vendor = dev.get_info<sycl::info::device::vendor>();
                auto const is_intel_gpu = dev.is_gpu() && (vendor.find("AMD") != std::string::npos);

                return is_intel_gpu ? 1 : -1;
            }
        };
    } // namespace detail

    //! The SYCL device manager.
    using PlatformGpuSyclAmd = PlatformGenericSycl<TagGpuSyclAmd>;
} // namespace aV2P

#endif

/* Copyright 2026 Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <aV2P/acc/Tag.hpp>
#include <aV2P/warp/WarpGenericSycl.hpp>

namespace aV2P::test
{
    //! Return why the warp operations are not supported by the accelerator TAcc, or nullptr if they are supported.
    //! The warp operations are always supported, except by some SYCL backends and oneAPI versions: see
    //! aV2P/warp/WarpGenericSycl.hpp .
    template<typename TAcc>
    constexpr auto warpUnsupportedReason() -> char const*
    {
#ifdef AV2P_ACC_SYCL_ENABLED
        if constexpr(aV2P::accMatchesTags<
                         TAcc,
                         aV2P::TagCpuSycl,
                         aV2P::TagGpuSyclIntel,
                         aV2P::TagGpuSyclNvidia,
                         aV2P::TagGpuSyclAmd,
                         aV2P::TagFpgaSyclIntel,
                         aV2P::TagGenericSycl>)
        {
            return aV2P::warp::detail::syclWarpUnsupportedReason<aV2P::AccToTag<TAcc>>();
        }
#endif
        return nullptr;
    }
} // namespace aV2P::test

/* Copyright 2026 Mykhailo Varvarin
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Common.hpp"
#include "aV2P/core/Interface.hpp"
#include "aV2P/grid/Traits.hpp"

#ifdef AV2P_ACC_SYCL_ENABLED

#    include <sycl/sycl.hpp>

namespace aV2P
{
    //! The SYCL grid synchronization entry point.
    //!
    //! In alpaka 2.2 this used the oneAPI root group barrier, which is only valid for a cooperative
    //! launch.  alpaka 3.0.0 does not expose cooperative launch, so the type and the trait entry point
    //! remain for source compatibility, and using them fails with a diagnostic instead of silently
    //! turning into a no-op.
    template<typename TDim>
    class GridSyncGenericSycl : public interface::Implements<ConceptGridSync, GridSyncGenericSycl<TDim>>
    {
    public:
        GridSyncGenericSycl(sycl::nd_item<TDim::value> work_item) : my_item{work_item}
        {
        }

        sycl::nd_item<TDim::value> my_item;
    };

    namespace trait
    {
        template<typename TDim>
        struct SyncGridThreads<GridSyncGenericSycl<TDim>>
        {
            template<typename TGridSync = GridSyncGenericSycl<TDim>>
            AV2P_NO_HOST_ACC_WARNING
            AV2P_FN_ACC static auto syncGridThreads([[maybe_unused]] TGridSync const& /*gridSync*/) -> void
            {
                static_assert(aV2P::detail::gridSyncUnsupported<TGridSync>, AV2P_GRID_SYNC_COOPERATIVE_MESSAGE);
            }
        };

    } // namespace trait

} // namespace aV2P

#endif

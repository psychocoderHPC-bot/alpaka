/* Copyright 2026 Mykhailo Varvarin
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Common.hpp"
#include "aV2P/core/Interface.hpp"
#include "aV2P/grid/Traits.hpp"

#ifdef AV2P_ACC_SYCL_ENABLED

namespace aV2P
{
    //! The SYCL grid synchronization entry point.
    //!
    //! In alpaka 2.2 this used the oneAPI root group barrier, which is only valid for a cooperative
    //! launch.  alpaka 3.0.0 does not expose cooperative launch, so the type and the trait entry point
    //! remain for source compatibility, and using them fails with a diagnostic instead of silently
    //! turning into a no-op.  The type deliberately stores no SYCL internals: a diagnostic-only entry
    //! point must not depend on a specific oneAPI implementation header.
    template<typename TDim>
    class GridSyncGenericSycl : public interface::Implements<ConceptGridSync, GridSyncGenericSycl<TDim>>
    {
    };

    namespace trait
    {
        template<typename TDim>
        struct SyncGridThreads<GridSyncGenericSycl<TDim>>
        {
            // No AV2P_NO_HOST_ACC_WARNING here: the diagnostic function does not call host code and the
            // pragma is only valid immediately before a declaration.
            template<typename TGridSync = GridSyncGenericSycl<TDim>>
            AV2P_FN_ACC static auto syncGridThreads([[maybe_unused]] TGridSync const& /*gridSync*/) -> void
            {
                static_assert(aV2P::detail::gridSyncUnsupported<TGridSync>, AV2P_GRID_SYNC_COOPERATIVE_MESSAGE);
            }
        };

    } // namespace trait

} // namespace aV2P

#endif

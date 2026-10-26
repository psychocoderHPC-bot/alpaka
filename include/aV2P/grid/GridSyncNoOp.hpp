/* Copyright 2026 Mykhailo Varvarin
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Common.hpp"
#include "aV2P/core/Interface.hpp"
#include "aV2P/grid/Traits.hpp"

namespace aV2P
{
    //! The NoOp grid synchronization for accelerators that only support a single thread with cooperative kernels.
    class GridSyncNoOp : public interface::Implements<ConceptGridSync, GridSyncNoOp>
    {
    };

    namespace trait
    {
        template<>
        struct SyncGridThreads<GridSyncNoOp>
        {
            AV2P_NO_HOST_ACC_WARNING
            AV2P_FN_ACC static auto syncGridThreads(GridSyncNoOp const& /*gridSync*/) -> void
            {
                // Nothing to do.
            }
        };

    } // namespace trait

} // namespace aV2P

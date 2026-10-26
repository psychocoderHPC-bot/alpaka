/* Copyright 2026 Mykhailo Varvarin
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Common.hpp"
#include "aV2P/core/Interface.hpp"
#include "aV2P/grid/Traits.hpp"

#ifdef _OPENMP

namespace aV2P
{
    //! The grid synchronization for OMP accelerators.
    class GridSyncOmp : public interface::Implements<ConceptGridSync, GridSyncOmp>
    {
    };

    namespace trait
    {
        template<>
        struct SyncGridThreads<GridSyncOmp>
        {
            AV2P_NO_HOST_ACC_WARNING
            AV2P_FN_ACC static auto syncGridThreads(GridSyncOmp const& /*gridSync*/) -> void
            {
#    pragma omp barrier
            }
        };

    } // namespace trait

} // namespace aV2P

#endif

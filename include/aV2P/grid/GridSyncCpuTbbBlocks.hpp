/* Copyright 2026 Mykhailo Varvarin
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/BarrierTbb.h"
#include "aV2P/core/Common.hpp"
#include "aV2P/core/Interface.hpp"
#include "aV2P/grid/Traits.hpp"

#ifdef AV2P_ACC_CPU_B_TBB_T_SEQ_ENABLED

namespace aV2P
{
    //! The thread id map barrier grid synchronization for TBB.
    template<typename TIdx>
    class GridSyncBarrierTbb : public interface::Implements<ConceptGridSync, GridSyncBarrierTbb<TIdx>>
    {
    public:
        using Barrier = core::tbb::BarrierThread<TIdx>;

        // Get reference to the barrier from the outside because we need it to be shared between blocks
        AV2P_FN_HOST explicit GridSyncBarrierTbb(Barrier& barrier)
        {
            m_barrier = &barrier;
        }

        Barrier* m_barrier;
    };

    namespace trait
    {
        template<typename TIdx>
        struct SyncGridThreads<GridSyncBarrierTbb<TIdx>>
        {
            AV2P_FN_HOST static auto syncGridThreads(GridSyncBarrierTbb<TIdx> const& gridSync) -> void
            {
                gridSync.m_barrier->wait();
            }
        };

    } // namespace trait
} // namespace aV2P

#endif

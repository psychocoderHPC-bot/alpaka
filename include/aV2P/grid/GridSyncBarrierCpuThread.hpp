/* Copyright 2026 Mykhailo Varvarin
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/BarrierThread.hpp"
#include "aV2P/core/Common.hpp"
#include "aV2P/core/Interface.hpp"
#include "aV2P/grid/Traits.hpp"

#include <thread>

#ifdef AV2P_ACC_CPU_B_SEQ_T_THREADS_ENABLED

namespace aV2P
{
    //! The thread id map barrier grid synchronization.
    template<typename TIdx>
    class GridSyncBarrierThread : public interface::Implements<ConceptGridSync, GridSyncBarrierThread<TIdx>>
    {
    public:
        using Barrier = core::threads::BarrierThread<TIdx>;

        AV2P_FN_HOST GridSyncBarrierThread(TIdx const& blockThreadCount) : m_barrier(blockThreadCount)
        {
        }

        Barrier mutable m_barrier;
    };

    namespace trait
    {
        template<typename TIdx>
        struct SyncGridThreads<GridSyncBarrierThread<TIdx>>
        {
            AV2P_FN_HOST static auto syncGridThreads(GridSyncBarrierThread<TIdx> const& gridSync) -> void
            {
                gridSync.m_barrier.wait();
            }
        };

    } // namespace trait
} // namespace aV2P

#endif

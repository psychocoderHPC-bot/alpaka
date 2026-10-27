/* Copyright 2026 Mykhailo Varvarin
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Common.hpp"
#include "aV2P/core/Interface.hpp"

//! Diagnostic used by every grid-synchronization entry point that needs a cooperative launch.
//!
//! A single message keeps the CUDA/HIP, oneAPI and accelerator-facade diagnostics identical.  The
//! implementations in this directory are diagnostic-only: alpaka 3.0.0 has no cooperative launch
//! and therefore cannot provide a grid-wide barrier on a GPU back-end.
#define AV2P_GRID_SYNC_COOPERATIVE_MESSAGE                                                                            \
    "aV2P: grid-wide synchronization requires a cooperative kernel launch. The pinned alpaka 3.0.0 "                  \
    "does not provide cooperative launch, so grid synchronization is not supported on this back-end. "                \
    "Use a CPU back-end (CpuSerial or CpuOmp2Blocks) for grid synchronization."

namespace aV2P
{
    namespace detail
    {
        //! Dependent false: makes the unsupported-backend diagnostics instantiate only on use.
        template<typename>
        inline constexpr bool gridSyncUnsupported = false;
    } // namespace detail

    struct ConceptGridSync
    {
    };

    //! The grid synchronization traits.
    namespace trait
    {
        //! The grid synchronization operation trait.
        template<typename TGridSync, typename TSfinae = void>
        struct SyncGridThreads;

    } // namespace trait

    //! Synchronizes all threads within the current grid. Works only for cooperative kernels.
    //! NOTE: when compiled with CUDA Clang default build type locks up if numberOfBlocks > 2 * multiProcessorCount.
    //! https://github.com/llvm/llvm-project/issues/98886
    //!
    //! \tparam TGridSync The grid synchronization implementation type.
    //! \param gridSync The grid synchronization implementation.
    AV2P_NO_HOST_ACC_WARNING
    template<typename TGridSync>
    AV2P_FN_ACC auto syncGridThreads(TGridSync const& gridSync) -> void
    {
        using ImplementationBase = interface::ImplementationBase<ConceptGridSync, TGridSync>;
        trait::SyncGridThreads<ImplementationBase>::syncGridThreads(gridSync);
    }

} // namespace aV2P

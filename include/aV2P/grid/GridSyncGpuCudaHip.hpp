/* Copyright 2026 Mykhailo Varvarin, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Common.hpp"
#include "aV2P/core/Config.hpp"
#include "aV2P/core/Interface.hpp"
#include "aV2P/grid/Traits.hpp"

#if defined(AV2P_ACC_GPU_CUDA_ENABLED) || defined(AV2P_ACC_GPU_HIP_ENABLED)

namespace aV2P
{
    //! The GPU CUDA/HIP grid synchronization entry point.
    //!
    //! In alpaka 2.2 this was implemented with `cooperative_groups::this_grid().sync()`.  A grid-wide
    //! barrier is only valid in a cooperative kernel launch, and alpaka 3.0.0 does not expose
    //! cooperative launch.  The type and the trait entry point are kept for source compatibility, but
    //! any instantiation that actually asks for the synchronization fails with a diagnostic instead of
    //! silently turning into a no-op.
    class GridSyncCudaHipBuiltIn : public interface::Implements<ConceptGridSync, GridSyncCudaHipBuiltIn>
    {
    };

#    if !defined(AV2P_HOST_ONLY)

#        if defined(AV2P_ACC_GPU_CUDA_ENABLED) && !AV2P_LANG_CUDA
#            error If AV2P_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#        endif

#        if defined(AV2P_ACC_GPU_HIP_ENABLED) && !AV2P_LANG_HIP
#            error If AV2P_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#        endif

    namespace trait
    {
        template<>
        struct SyncGridThreads<GridSyncCudaHipBuiltIn>
        {
            template<typename TGridSync = GridSyncCudaHipBuiltIn>
            AV2P_NO_HOST_ACC_WARNING
            AV2P_FN_ACC static auto syncGridThreads([[maybe_unused]] TGridSync const& /*gridSync*/) -> void
            {
                static_assert(aV2P::detail::gridSyncUnsupported<TGridSync>, AV2P_GRID_SYNC_COOPERATIVE_MESSAGE);
            }
        };

    } // namespace trait

#    endif

} // namespace aV2P

#endif

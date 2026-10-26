/* Copyright 2026 Mykhailo Varvarin, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Common.hpp"
#include "aV2P/core/Config.hpp"
#include "aV2P/core/Interface.hpp"
#include "aV2P/grid/Traits.hpp"

#if defined(AV2P_ACC_GPU_CUDA_ENABLED) || defined(AV2P_ACC_GPU_HIP_ENABLED)

#    if defined(AV2P_ACC_GPU_CUDA_ENABLED)
// The cooperative groups headers from CUDA 12.0.0 use PTX register names with a single '%' in an inline asm statement
// with operands, which clang rejects ("invalid % escape in inline assembly string"). The asm is only used for the
// multi-warp scratch space in the reserved shared memory, which the grid synchronization does not need, so disable it.
#        if AV2P_COMP_CLANG_CUDA && defined(CUDART_VERSION) && (CUDART_VERSION == 12000)                            \
            && !defined(_CG_USER_PROVIDED_SHARED_MEMORY)
#            pragma clang diagnostic push
#            pragma clang diagnostic ignored "-Wreserved-macro-identifier"
#            define _CG_USER_PROVIDED_SHARED_MEMORY
#            pragma clang diagnostic pop
#        endif
#        include <cooperative_groups.h>
#    endif

#    if defined(AV2P_ACC_GPU_HIP_ENABLED)
#        include <hip/hip_cooperative_groups.h>
#    endif


namespace aV2P
{
    //! The GPU CUDA/HIP grid synchronization.
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
            __device__ static auto syncGridThreads(GridSyncCudaHipBuiltIn const& /*gridSync*/) -> void
            {
                cooperative_groups::this_grid().sync();
            }
        };

    } // namespace trait

#    endif

} // namespace aV2P

#endif

/* Copyright 2022 Benjamin Worpitz, Matthias Werner, Andrea Bocci, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/block/sync/Traits.hpp"
#include "aV2P/core/Config.hpp"
#include "aV2P/core/Interface.hpp"

#if defined(AV2P_ACC_GPU_CUDA_ENABLED) || defined(AV2P_ACC_GPU_HIP_ENABLED)

namespace aV2P
{
    //! The GPU CUDA/HIP block synchronization.
    class BlockSyncUniformCudaHipBuiltIn
        : public interface::Implements<ConceptBlockSync, BlockSyncUniformCudaHipBuiltIn>
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
        struct SyncBlockThreads<BlockSyncUniformCudaHipBuiltIn>
        {
            __device__ static auto syncBlockThreads(BlockSyncUniformCudaHipBuiltIn const& /*blockSync*/) -> void
            {
                __syncthreads();
            }
        };

        template<>
        struct SyncBlockThreadsPredicate<BlockCount, BlockSyncUniformCudaHipBuiltIn>
        {
            __device__ static auto syncBlockThreadsPredicate(
                BlockSyncUniformCudaHipBuiltIn const& /*blockSync*/,
                int predicate) -> int
            {
#        if defined(__HIP_ARCH_HAS_SYNC_THREAD_EXT__) && __HIP_ARCH_HAS_SYNC_THREAD_EXT__ == 0 && AV2P_COMP_HIP
#            if AV2P_COMP_CLANG >= AV2P_VERSION_NUMBER(21, 0, 0)
#                pragma clang diagnostic push
#                pragma clang diagnostic ignored "-Wunique-object-duplication"
#            endif
                // workaround for unsupported syncthreads_* operation on AMD hardware without sync extension
                __shared__ int tmp;
#            if AV2P_COMP_CLANG >= AV2P_VERSION_NUMBER(21, 0, 0)
#                pragma clang diagnostic pop
#            endif
                __syncthreads();
                if(threadIdx.x == 0)
                    tmp = 0;
                __syncthreads();
                if(predicate)
                    ::atomicAdd(&tmp, 1);
                __syncthreads();

                return tmp;
#        else
                return __syncthreads_count(predicate);
#        endif
            }
        };

        template<>
        struct SyncBlockThreadsPredicate<BlockAnd, BlockSyncUniformCudaHipBuiltIn>
        {
            __device__ static auto syncBlockThreadsPredicate(
                BlockSyncUniformCudaHipBuiltIn const& /*blockSync*/,
                int predicate) -> int
            {
#        if defined(__HIP_ARCH_HAS_SYNC_THREAD_EXT__) && __HIP_ARCH_HAS_SYNC_THREAD_EXT__ == 0 && AV2P_COMP_HIP
#            if AV2P_COMP_CLANG >= AV2P_VERSION_NUMBER(21, 0, 0)
#                pragma clang diagnostic push
#                pragma clang diagnostic ignored "-Wunique-object-duplication"
#            endif
                // workaround for unsupported syncthreads_* operation on AMD hardware without sync extension
                __shared__ int tmp;
#            if AV2P_COMP_CLANG >= AV2P_VERSION_NUMBER(21, 0, 0)
#                pragma clang diagnostic pop
#            endif
                __syncthreads();
                if(threadIdx.x == 0)
                    tmp = 1;
                __syncthreads();
                if(!predicate)
                    ::atomicAnd(&tmp, 0);
                __syncthreads();

                return tmp;
#        else
                return __syncthreads_and(predicate);
#        endif
            }
        };

        template<>
        struct SyncBlockThreadsPredicate<BlockOr, BlockSyncUniformCudaHipBuiltIn>
        {
            __device__ static auto syncBlockThreadsPredicate(
                BlockSyncUniformCudaHipBuiltIn const& /*blockSync*/,
                int predicate) -> int
            {
#        if defined(__HIP_ARCH_HAS_SYNC_THREAD_EXT__) && __HIP_ARCH_HAS_SYNC_THREAD_EXT__ == 0 && AV2P_COMP_HIP
#            if AV2P_COMP_CLANG >= AV2P_VERSION_NUMBER(21, 0, 0)
#                pragma clang diagnostic push
#                pragma clang diagnostic ignored "-Wunique-object-duplication"
#            endif
                // workaround for unsupported syncthreads_* operation on AMD hardware without sync extension
                __shared__ int tmp;
#            if AV2P_COMP_CLANG >= AV2P_VERSION_NUMBER(21, 0, 0)
#                pragma clang diagnostic pop
#            endif
                __syncthreads();
                if(threadIdx.x == 0)
                    tmp = 0;
                __syncthreads();
                if(predicate)
                    ::atomicOr(&tmp, 1);
                __syncthreads();

                return tmp;
#        else
                return __syncthreads_or(predicate);
#        endif
            }
        };
    } // namespace trait

#    endif

} // namespace aV2P

#endif

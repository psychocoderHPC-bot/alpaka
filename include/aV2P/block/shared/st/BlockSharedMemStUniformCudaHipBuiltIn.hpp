/* Copyright 2022 Benjamin Worpitz, Erik Zenker, René Widera, Matthias Werner, Andrea Bocci, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/block/shared/st/Traits.hpp"
#include "aV2P/core/Config.hpp"
#include "aV2P/core/Interface.hpp"

#include <cstdint>
#include <type_traits>

#if defined(AV2P_ACC_GPU_CUDA_ENABLED) || defined(AV2P_ACC_GPU_HIP_ENABLED)

namespace aV2P
{
    //! The GPU CUDA/HIP block shared memory allocator.
    class BlockSharedMemStUniformCudaHipBuiltIn
        : public interface::Implements<ConceptBlockSharedSt, BlockSharedMemStUniformCudaHipBuiltIn>
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
        template<typename T, std::size_t TuniqueId>
        struct DeclareSharedVar<T, TuniqueId, BlockSharedMemStUniformCudaHipBuiltIn>
        {
            __device__ static auto declareVar(BlockSharedMemStUniformCudaHipBuiltIn const&) -> T&
            {
                __shared__ uint8_t shMem alignas(alignof(T))[sizeof(T)];
                return *(reinterpret_cast<T*>(shMem));
            }
        };

        template<>
        struct FreeSharedVars<BlockSharedMemStUniformCudaHipBuiltIn>
        {
            __device__ static auto freeVars(BlockSharedMemStUniformCudaHipBuiltIn const&) -> void
            {
                // Nothing to do. CUDA/HIP block shared memory is automatically freed when all threads left the block.
            }
        };
    } // namespace trait

#    endif

} // namespace aV2P

#endif

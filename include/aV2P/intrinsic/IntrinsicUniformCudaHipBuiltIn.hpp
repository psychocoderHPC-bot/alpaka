/* Copyright 2022 Sergei Bastrakov, Andrea Bocci, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Config.hpp"
#include "aV2P/core/Interface.hpp"
#include "aV2P/intrinsic/Traits.hpp"

#if defined(AV2P_ACC_GPU_CUDA_ENABLED) || defined(AV2P_ACC_GPU_HIP_ENABLED)

namespace aV2P
{
    //! The GPU CUDA/HIP intrinsic.
    class IntrinsicUniformCudaHipBuiltIn
        : public interface::Implements<ConceptIntrinsic, IntrinsicUniformCudaHipBuiltIn>
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
        struct Popcount<IntrinsicUniformCudaHipBuiltIn>
        {
            __device__ static auto popcount(IntrinsicUniformCudaHipBuiltIn const& /*intrinsic*/, std::uint32_t value)
                -> std::int32_t
            {
                // clang as CUDA compiler change the interface to unsigned values for clang >=18
#        if AV2P_COMP_CLANG && AV2P_LANG_CUDA && AV2P_COMP_CLANG < AV2P_VERSION_NUMBER(18, 0, 0)
                return __popc(static_cast<int>(value));
#        else
                return static_cast<std::int32_t>(__popc(static_cast<unsigned int>(value)));
#        endif
            }

            __device__ static auto popcount(IntrinsicUniformCudaHipBuiltIn const& /*intrinsic*/, std::uint64_t value)
                -> std::int32_t
            {
                // clang as CUDA compiler change the interface to unsigned values for clang >=18
#        if AV2P_COMP_CLANG && AV2P_LANG_CUDA && AV2P_COMP_CLANG < AV2P_VERSION_NUMBER(18, 0, 0)
                return __popcll(static_cast<long long>(value));
#        else
                return static_cast<std::int32_t>(__popcll(static_cast<unsigned long long>(value)));
#        endif
            }
        };

        template<>
        struct Ffs<IntrinsicUniformCudaHipBuiltIn>
        {
            __device__ static auto ffs(IntrinsicUniformCudaHipBuiltIn const& /*intrinsic*/, std::int32_t value)
                -> std::int32_t
            {
                return static_cast<std::int32_t>(__ffs(static_cast<int>(value)));
            }

            __device__ static auto ffs(IntrinsicUniformCudaHipBuiltIn const& /*intrinsic*/, std::int64_t value)
                -> std::int32_t
            {
                return static_cast<std::int32_t>(__ffsll(static_cast<long long>(value)));
            }
        };
    } // namespace trait

#    endif

} // namespace aV2P

#endif

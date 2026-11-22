/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/common.hpp"

#include <cstdint>
#include <type_traits>

#if defined(__SSE4_1__) && (defined(__i386__) || defined(__x86_64__))
#    include <smmintrin.h>
#endif

namespace alpaka::internal
{
    /** Whether a streaming (non-temporal / evict-first) 16 byte load is available for a SIMD pack.
     *
     * The helper is only enabled when the intrinsic exists for the current compiler pass, therefore the caller can
     * fall back to the plain load and the generated code stays unchanged if the hint is not supported.
     */
    template<typename T_Type, uint32_t T_width>
    inline constexpr bool streamLoadAvailable()
    {
        if constexpr(sizeof(T_Type) * T_width != 16u)
        {
            return false;
        }
        else
        {
#if ALPAKA_LANG_HIP && defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__ == 1
#    if defined(__has_builtin) && __has_builtin(__builtin_nontemporal_load)
            return true;
#    else
            return false;
#    endif
#elif ALPAKA_LANG_CUDA && defined(__CUDA_ARCH__)
            return true;
#elif ALPAKA_LANG_SYCL && defined(__SYCL_DEVICE_ONLY__)
#    if defined(__has_builtin) && __has_builtin(__builtin_nontemporal_load)
            return true;
#    else
            return false;
#    endif
#else
#    if defined(__has_builtin) && __has_builtin(__builtin_nontemporal_load)
            return true;
#    elif defined(__SSE4_1__)
            return true;
#    else
            return false;
#    endif
#endif
        }
    }

    /** Perform a 16 byte streaming load of T_width elements from src into the 16 byte aligned dst.
     *
     * Only call this function if streamLoadAvailable<T_Type, T_width>() is true.
     */
    template<typename T_Type, uint32_t T_width>
    ALPAKA_FN_ACC constexpr void streamLoad(T_Type* dst, T_Type const* src)
    {
        static_assert(sizeof(T_Type) * T_width == 16u);
#if ALPAKA_LANG_HIP && defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__ == 1
        // The HIP builtin rejects the SIMD struct types, a vector reinterpretation of the 16 byte pack is accepted.
        using NativeVec = T_Type __attribute__((vector_size(16)));
        *reinterpret_cast<NativeVec*>(dst) = __builtin_nontemporal_load(reinterpret_cast<NativeVec const*>(src));
#elif ALPAKA_LANG_CUDA && defined(__CUDA_ARCH__)
        // __ldcg has no generic overload, use the matching 16 byte CUDA vector type for the raw bit copy.
        // It bypasses L1 and keeps L2, which is the closest CUDA equivalent of AMD's evict-first policy. Measured on
        // sm_80 __ldcs (evict-first in L2 as well) is slower than the plain load for all BabelStream cells.
        using NativeVec = std::conditional_t<
            std::is_floating_point_v<T_Type> && sizeof(T_Type) == 8u,
            double2,
            std::conditional_t<std::is_floating_point_v<T_Type>, float4, uint4>>;
        *reinterpret_cast<NativeVec*>(dst) = __ldcg(reinterpret_cast<NativeVec const*>(src));
#elif ALPAKA_LANG_SYCL && defined(__SYCL_DEVICE_ONLY__)
        using NativeVec = T_Type __attribute__((vector_size(16)));
        *reinterpret_cast<NativeVec*>(dst) = __builtin_nontemporal_load(reinterpret_cast<NativeVec const*>(src));
#else
#    if defined(__has_builtin) && __has_builtin(__builtin_nontemporal_load)
        using NativeVec = T_Type __attribute__((vector_size(16)));
        *reinterpret_cast<NativeVec*>(dst) = __builtin_nontemporal_load(reinterpret_cast<NativeVec const*>(src));
#    elif defined(__SSE4_1__)
        // movntdqa is only a hint: on normal write-back memory it behaves like a regular load.
        _mm_store_si128(
            reinterpret_cast<__m128i*>(dst),
            _mm_stream_load_si128(reinterpret_cast<__m128i*>(const_cast<T_Type*>(src))));
#    endif
#endif
    }
} // namespace alpaka::internal

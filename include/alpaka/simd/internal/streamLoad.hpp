/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/common.hpp"
#include "alpaka/mem/Alignment.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>

#if defined(__SSE4_1__) && (defined(__i386__) || defined(__x86_64__))
#    include <smmintrin.h>
#    if defined(__AVX2__)
#        include <immintrin.h>
#    endif
#endif

namespace alpaka::internal
{
    /** Operand width in byte of one 16 byte cache-hinted load instruction.
     *
     * The value is a property of the instruction, not of the SIMD pack: CUDA's ``__ldcg`` is declared for the 16 byte
     * vector types only and x86's ``movntdqa`` moves 16 byte with SSE4.1 and 32 byte with AVX2. A pack wider than one
     * operand is covered by several chunks, a pack narrower than one operand is loaded without the hint.
     */
    inline constexpr size_t intrinsicStreamLoadWidthInByte = 16u;

    /** Byte width of one cache-hinted load instruction for this compile pass and value type.
     *
     * - CUDA: ``__ldcg`` exists for the 16 byte vector types only, therefore one instruction moves 16 byte.
     * - x86 with SSE4.1: ``movntdqa`` moves 16 byte, its AVX2 form 32 byte.
     * - HIP, SYCL and non-x86 hosts: ``__builtin_nontemporal_load`` accepts a vector of any width, so one load covers
     *   the whole pack.
     *
     * @return 0 if the compile pass has no cache-hinted load.
     */
    template<typename T_Type, uint32_t T_width>
    consteval size_t streamLoadInstructionWidth()
    {
#if ALPAKA_LANG_HIP && defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__ == 1
#    if defined(__has_builtin) && __has_builtin(__builtin_nontemporal_load)
        return sizeof(T_Type) * T_width;
#    else
        return 0u;
#    endif
#elif ALPAKA_LANG_CUDA && defined(__CUDA_ARCH__)
        return intrinsicStreamLoadWidthInByte;
#elif ALPAKA_LANG_SYCL && defined(__SYCL_DEVICE_ONLY__)
#    if defined(__has_builtin) && __has_builtin(__builtin_nontemporal_load)
        return sizeof(T_Type) * T_width;
#    else
        return 0u;
#    endif
#elif defined(__SSE4_1__) && (defined(__i386__) || defined(__x86_64__))
#    if defined(__AVX2__)
        return 2u * intrinsicStreamLoadWidthInByte;
#    else
        return intrinsicStreamLoadWidthInByte;
#    endif
#elif defined(__has_builtin) && __has_builtin(__builtin_nontemporal_load)
        return sizeof(T_Type) * T_width;
#else
        return 0u;
#endif
    }

    /** Chunk width in byte of one hinted load, 0 if the hint must not be applied.
     *
     * The chunk has to divide the pack size, so an integer number of instructions covers the pack, and it has to
     * divide the alignment which the caller promised, so every chunk starts on an aligned address. The widest
     * instruction which satisfies both is used, below the smallest instruction width the hint is dropped. Both
     * inputs are derived from the SIMD type and from the alignment type; the pack width is never assumed.
     */
    template<typename T_Type, uint32_t T_width, alpaka::concepts::Alignment T_Alignment>
    consteval size_t streamLoadChunkWidth()
    {
        constexpr size_t packWidth = sizeof(T_Type) * T_width;
        constexpr size_t alignment = T_Alignment::template get<T_Type>();
        // Start at the largest power of two which the instruction can move, then try every narrower power of two.
        size_t chunkWidth = std::bit_floor(streamLoadInstructionWidth<T_Type, T_width>());
        while(chunkWidth >= intrinsicStreamLoadWidthInByte)
        {
            if(packWidth % chunkWidth == 0u && alignment % chunkWidth == 0u)
                return chunkWidth;
            chunkWidth /= 2u;
        }
        return 0u;
    }

    /** Whether a streaming (non-temporal / evict-first) load can cover the SIMD pack.
     *
     * The check depends on the value type, the SIMD width and the alignment the caller promised, therefore a caller
     * can fall back to the plain load and the generated code stays unchanged if the hint is not supported.
     */
    template<typename T_Type, uint32_t T_width, alpaka::concepts::Alignment T_Alignment>
    inline constexpr bool streamLoadAvailable()
    {
        return streamLoadChunkWidth<T_Type, T_width, T_Alignment>() != 0u;
    }

    /** Perform the cache-hinted load of T_width elements from src into dst.
     *
     * The pack is covered by streamLoadChunkWidth() byte chunks. The chunk width divides the alignment promised by
     * T_Alignment, therefore src and dst, which the caller aligned for the pack, are aligned for every chunk.
     *
     * Only call this function if streamLoadAvailable<T_Type, T_width, T_Alignment>() is true.
     */
    template<typename T_Type, uint32_t T_width, alpaka::concepts::Alignment T_Alignment>
    ALPAKA_FN_ACC constexpr void streamLoad(T_Type* dst, T_Type const* src)
    {
        constexpr size_t chunkWidth = streamLoadChunkWidth<T_Type, T_width, T_Alignment>();
        constexpr size_t packWidth = sizeof(T_Type) * T_width;
        static_assert(chunkWidth != 0u, "No cache-hinted load is available for this type or alignment.");
        static_assert(
            packWidth % chunkWidth == 0u,
            "A hinted load must cover the SIMD pack with an integer number of chunks.");
        constexpr uint32_t numChunks = static_cast<uint32_t>(packWidth / chunkWidth);

#if ALPAKA_LANG_HIP && defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__ == 1
        // The HIP builtin rejects the SIMD struct types, a vector reinterpretation of one chunk is accepted.
        using NativeVec = T_Type __attribute__((vector_size(chunkWidth)));
        for(uint32_t i = 0u; i < numChunks; ++i)
            reinterpret_cast<NativeVec*>(dst)[i]
                = __builtin_nontemporal_load(reinterpret_cast<NativeVec const*>(src) + i);
#elif ALPAKA_LANG_CUDA && defined(__CUDA_ARCH__)
        // __ldcg has no generic overload and is limited to the 16 byte vector types, therefore the widest possible
        // pack is 16 byte. uint4 holds the raw bits of one chunk for every element type, so no element type dispatch
        // is required. It bypasses L1 and keeps L2, the closest CUDA equivalent of AMD's evict-first policy. Measured
        // on sm_80, __ldcs (evict-first in L2 as well) is slower than the plain load for all BabelStream cells.
        static_assert(chunkWidth == intrinsicStreamLoadWidthInByte);
        using Pack = uint4;
        static_assert(sizeof(Pack) == intrinsicStreamLoadWidthInByte);
        for(uint32_t i = 0u; i < numChunks; ++i)
            reinterpret_cast<Pack*>(dst)[i] = __ldcg(reinterpret_cast<Pack const*>(src) + i);
#elif ALPAKA_LANG_SYCL && defined(__SYCL_DEVICE_ONLY__)
        using NativeVec = T_Type __attribute__((vector_size(chunkWidth)));
        for(uint32_t i = 0u; i < numChunks; ++i)
            reinterpret_cast<NativeVec*>(dst)[i]
                = __builtin_nontemporal_load(reinterpret_cast<NativeVec const*>(src) + i);
#elif defined(__SSE4_1__) && (defined(__i386__) || defined(__x86_64__))
#    if defined(__AVX2__)
        if constexpr(chunkWidth == 2u * intrinsicStreamLoadWidthInByte)
        {
            for(uint32_t i = 0u; i < numChunks; ++i)
                _mm256_store_si256(
                    reinterpret_cast<__m256i*>(dst) + i,
                    _mm256_stream_load_si256(reinterpret_cast<__m256i*>(const_cast<T_Type*>(src)) + i));
        }
        else
#    endif
        {
            static_assert(chunkWidth == intrinsicStreamLoadWidthInByte);
            // movntdqa is only a hint: on normal write-back memory it behaves like a regular load.
            for(uint32_t i = 0u; i < numChunks; ++i)
                _mm_store_si128(
                    reinterpret_cast<__m128i*>(dst) + i,
                    _mm_stream_load_si128(reinterpret_cast<__m128i*>(const_cast<T_Type*>(src)) + i));
        }
#elif defined(__has_builtin) && __has_builtin(__builtin_nontemporal_load)
        using NativeVec = T_Type __attribute__((vector_size(chunkWidth)));
        for(uint32_t i = 0u; i < numChunks; ++i)
            reinterpret_cast<NativeVec*>(dst)[i]
                = __builtin_nontemporal_load(reinterpret_cast<NativeVec const*>(src) + i);
#endif
    }
} // namespace alpaka::internal

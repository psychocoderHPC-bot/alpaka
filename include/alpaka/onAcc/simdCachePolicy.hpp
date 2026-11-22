/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/common.hpp"

namespace alpaka::onAcc
{
    /** Cache policy for SIMD loads.
     *
     * A policy is a compile-time tag passed to the load where the memory is accessed, for example
     * ``simdPtr.load(simdCachePolicy::Stream{})``. Therefore a kernel can select the policy per access and one
     * algorithm call can mix inputs with different policies. The policy only changes how SIMD packs are loaded,
     * stores are untouched.
     */
    namespace simdCachePolicy
    {
        /** Use the default load instructions without a cache hint.
         *
         * This is the policy of a load without an argument and generates the previous code.
         */
        struct Default
        {
        };

        /** Evict-first/non-temporal loads for data which is not reused before it is evicted.
         *
         * The load width and the alignment come from the SIMD type and the alignment of the data source. A pack
         * which is a multiple of the instruction width is covered by several hinted loads, a pack below the
         * instruction width or a pointer which is not aligned for the instruction keeps the plain load. Stores are
         * untouched. Kernels which read data several times should stay on Default. The implementation uses
         * ``__builtin_nontemporal_load`` on HIP and SYCL, ``__ldcg`` (L1 bypass, L2 kept) on CUDA and
         * ``_mm_stream_load_si128`` or ``_mm256_stream_load_si256`` on the host, each behind an availability check.
         */
        struct Stream
        {
        };
    } // namespace simdCachePolicy
} // namespace alpaka::onAcc

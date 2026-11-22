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
         * The hint is applied to fully aligned 16 byte SIMD loads only; all other loads and every store use the
         * default path. Kernels which read data several times should stay on Default. The implementation uses
         * ``__builtin_nontemporal_load`` on HIP and SYCL, ``__ldcg`` (L1 bypass, L2 kept) on CUDA and
         * ``__builtin_nontemporal_load`` or ``_mm_stream_load_si128`` on the host, each behind an availability check.
         */
        struct Stream
        {
        };
    } // namespace simdCachePolicy
} // namespace alpaka::onAcc

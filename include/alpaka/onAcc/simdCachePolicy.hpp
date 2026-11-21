/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

namespace alpaka::onAcc
{
    /** Cache policy for SIMD memory accesses.
     *
     * The policy is a compile-time tag which is forwarded to the SIMD load implementation.
     * It is selected per algorithm call and does not change the behaviour of code which uses the
     * default policy.
     */
    namespace simdCachePolicy
    {
        /** Use the default load instructions without a cache hint. */
        struct Default
        {
        };

        /** Evict-first/non-temporal loads for data which is not reused.
         *
         * The measured gain was obtained together with one SIMD pack per call (``concurrent<16>()``), the hint alone
         * recovers less. Stores are not affected, and the hint is implemented for HIP/AMD only while every other
         * backend generates the same code as Default.
         */
        struct Stream
        {
        };
    } // namespace simdCachePolicy
} // namespace alpaka::onAcc

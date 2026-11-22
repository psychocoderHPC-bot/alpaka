/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/common.hpp"

#include <type_traits>
#include <utility>

namespace alpaka::onAcc
{
    /** Cache policy for SIMD memory accesses.
     *
     * A policy is a compile-time tag attached to a data source, it is never selected per algorithm call.
     * It only changes how SIMD packs are loaded, stores are untouched, and it does not change the behaviour of code
     * which uses an unannotated data source.
     */
    namespace simdCachePolicy
    {
        /** Use the default load instructions without a cache hint. */
        struct Default
        {
        };

        /** Evict-first/non-temporal loads for data which is not reused before it is evicted.
         *
         * The hint is applied to fully aligned 16 byte SIMD loads only; all other loads and every store use the
         * default path. Kernels which read data several times should stay on Default. The implementation uses
         * ``__builtin_nontemporal_load`` on HIP and SYCL, ``__ldcs`` on CUDA and ``__builtin_nontemporal_load`` or
         * ``_mm_stream_load_si128`` on the host, each behind an availability check.
         */
        struct Stream
        {
        };
    } // namespace simdCachePolicy
} // namespace alpaka::onAcc

namespace alpaka::trait
{
    /** Derive the SIMD cache policy carried by a data source type.
     *
     * Types without a nested ``CachePolicy`` alias fall back to ``simdCachePolicy::Default``.
     */
    template<typename T, typename = void>
    struct CachePolicyOf
    {
        using type = onAcc::simdCachePolicy::Default;
    };

    template<typename T>
    struct CachePolicyOf<T, std::void_t<typename T::CachePolicy>>
    {
        using type = typename T::CachePolicy;
    };
} // namespace alpaka::trait

namespace alpaka
{
    /** The cache policy of a data source, ``simdCachePolicy::Default`` if it does not carry one. */
    template<typename T>
    using CachePolicyOf_t = typename trait::CachePolicyOf<std::remove_cvref_t<T>>::type;

    /** Data source adapter which carries a SIMD cache policy.
     *
     * The adapter publicly derives from the original data source, therefore every accessor, the extents, the
     * alignment and the ``IDataSource``/``IMdSpan`` concepts keep working unchanged. The policy is exposed as the
     * nested ``CachePolicy`` alias so that ``SimdPtr`` can pick it up when it is constructed.
     *
     * @tparam T_View data source the policy is attached to
     * @tparam T_Policy cache policy, see ``simdCachePolicy``
     */
    template<typename T_View, typename T_Policy>
    struct WithCachePolicy : T_View
    {
        using CachePolicy = T_Policy;

        using T_View::T_View;

        ALPAKA_FN_HOST_ACC constexpr WithCachePolicy(T_View const& view) : T_View{view}
        {
        }

        ALPAKA_FN_HOST_ACC constexpr WithCachePolicy(T_View&& view) : T_View{std::move(view)}
        {
        }
    };

    /** Attach a cache policy to a data source: ``withCachePolicy(simdCachePolicy::Stream{}, view)``. */
    struct WithCachePolicyFn
    {
        template<typename T_Policy, typename T_View>
        ALPAKA_FN_HOST_ACC constexpr auto operator()(T_Policy, T_View&& view) const
        {
            return WithCachePolicy<std::remove_cvref_t<T_View>, T_Policy>{std::forward<T_View>(view)};
        }
    };

    inline constexpr WithCachePolicyFn withCachePolicy{};
} // namespace alpaka

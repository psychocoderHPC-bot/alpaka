/* Copyright 2023 Jan Stephan, Andrea Bocci, Luca Ferragina
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/atomic/Op.hpp"
#include "aV2P/atomic/Traits.hpp"
#include "aV2P/core/Positioning.hpp"
#include "aV2P/meta/DependentFalseType.hpp"

#include <cstdint>
#include <type_traits>

#ifdef AV2P_ACC_SYCL_ENABLED

#    include <sycl/sycl.hpp>

namespace aV2P
{
    //! The SYCL accelerator atomic ops.
    //
    //  Atomics can used in the hierarchy level grids, blocks and threads.
    //  Atomics are not guaranteed to be safe between devices
    class AtomicGenericSycl
    {
    };

    namespace detail
    {
        template<typename THierarchy>
        struct SyclMemoryScope
        {
        };

        template<>
        struct SyclMemoryScope<hierarchy::Grids>
        {
            static constexpr auto value = sycl::memory_scope::device;
        };

        template<>
        struct SyclMemoryScope<hierarchy::Blocks>
        {
            static constexpr auto value = sycl::memory_scope::device;
        };

        template<>
        struct SyclMemoryScope<hierarchy::Threads>
        {
            static constexpr auto value = sycl::memory_scope::work_group;
        };

        template<typename T, typename THierarchy>
        using sycl_atomic_ref = sycl::atomic_ref<T, sycl::memory_order::relaxed, SyclMemoryScope<THierarchy>::value>;

        template<typename THierarchy, typename T, typename TOp>
        inline auto callAtomicOp(T* const addr, TOp&& op)
        {
            auto ref = sycl_atomic_ref<T, THierarchy>{*addr};
            return op(ref);
        }

        template<typename TRef, typename T, typename TEval>
        inline auto casWithCondition(T* const addr, TEval&& eval)
        {
            auto ref = TRef{*addr};
            auto old_val = ref.load();

            // prefer compare_exchange_weak when in a loop, assuming that eval is not expensive
            while(!ref.compare_exchange_weak(old_val, eval(old_val)))
            {
            }

            return old_val;
        }
    } // namespace detail
} // namespace aV2P

namespace aV2P::trait
{
    // Add.
    //! The SYCL accelerator atomic operation.
    template<typename T, typename THierarchy>
    struct AtomicOp<AtomicAdd, AtomicGenericSycl, T, THierarchy>
    {
        static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>, "SYCL atomics do not support this type");

        static auto atomicOp(AtomicGenericSycl const&, T* const addr, T const& value) -> T
        {
            return aV2P::detail::callAtomicOp<THierarchy>(
                addr,
                [&value](auto& ref) { return ref.fetch_add(value); });
        }
    };

    // Sub.
    //! The SYCL accelerator atomic operation.
    template<typename T, typename THierarchy>
    struct AtomicOp<AtomicSub, AtomicGenericSycl, T, THierarchy>
    {
        static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>, "SYCL atomics do not support this type");

        static auto atomicOp(AtomicGenericSycl const&, T* const addr, T const& value) -> T
        {
            return aV2P::detail::callAtomicOp<THierarchy>(
                addr,
                [&value](auto& ref) { return ref.fetch_sub(value); });
        }
    };

    // Min.
    //! The SYCL accelerator atomic operation.
    template<typename T, typename THierarchy>
    struct AtomicOp<AtomicMin, AtomicGenericSycl, T, THierarchy>
    {
        static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>, "SYCL atomics do not support this type");

        static auto atomicOp(AtomicGenericSycl const&, T* const addr, T const& value) -> T
        {
            return aV2P::detail::callAtomicOp<THierarchy>(
                addr,
                [&value](auto& ref) { return ref.fetch_min(value); });
        }
    };

    // Max.
    //! The SYCL accelerator atomic operation.
    template<typename T, typename THierarchy>
    struct AtomicOp<AtomicMax, AtomicGenericSycl, T, THierarchy>
    {
        static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>, "SYCL atomics do not support this type");

        static auto atomicOp(AtomicGenericSycl const&, T* const addr, T const& value) -> T
        {
            return aV2P::detail::callAtomicOp<THierarchy>(
                addr,
                [&value](auto& ref) { return ref.fetch_max(value); });
        }
    };

    // Exch.
    //! The SYCL accelerator atomic operation.
    template<typename T, typename THierarchy>
    struct AtomicOp<AtomicExch, AtomicGenericSycl, T, THierarchy>
    {
        static_assert(
            (std::is_integral_v<T> || std::is_floating_point_v<T>) and(sizeof(T) == 4 || sizeof(T) == 8),
            "SYCL atomics do not support this type");

        static auto atomicOp(AtomicGenericSycl const&, T* const addr, T const& value) -> T
        {
            return aV2P::detail::callAtomicOp<THierarchy>(addr, [&value](auto& ref) { return ref.exchange(value); });
        }
    };

    // Inc.
    //! The SYCL accelerator atomic operation.
    template<typename T, typename THierarchy>
    struct AtomicOp<AtomicInc, AtomicGenericSycl, T, THierarchy>
    {
        static_assert(
            std::is_unsigned_v<T> && (sizeof(T) == 4 || sizeof(T) == 8),
            "SYCL atomics support only 32- and 64-bits unsigned integral types");

        static auto atomicOp(AtomicGenericSycl const&, T* const addr, T const& value) -> T
        {
            auto inc = [&value](auto old_val)
            { return (old_val >= value) ? static_cast<T>(0) : (old_val + static_cast<T>(1)); };
            return aV2P::detail::casWithCondition<aV2P::detail::sycl_atomic_ref<T, THierarchy>>(addr, inc);
        }
    };

    // Dec.
    //! The SYCL accelerator atomic operation.
    template<typename T, typename THierarchy>
    struct AtomicOp<AtomicDec, AtomicGenericSycl, T, THierarchy>
    {
        static_assert(
            std::is_unsigned_v<T> && (sizeof(T) == 4 || sizeof(T) == 8),
            "SYCL atomics support only 32- and 64-bits unsigned integral types");

        static auto atomicOp(AtomicGenericSycl const&, T* const addr, T const& value) -> T
        {
            auto dec = [&value](auto& old_val)
            { return ((old_val == 0) || (old_val > value)) ? value : (old_val - static_cast<T>(1)); };
            return aV2P::detail::casWithCondition<aV2P::detail::sycl_atomic_ref<T, THierarchy>>(addr, dec);
        }
    };

    // And.
    //! The SYCL accelerator atomic operation.
    template<typename T, typename THierarchy>
    struct AtomicOp<AtomicAnd, AtomicGenericSycl, T, THierarchy>
    {
        static_assert(std::is_integral_v<T>, "Bitwise operations only supported for integral types.");

        static auto atomicOp(AtomicGenericSycl const&, T* const addr, T const& value) -> T
        {
            return aV2P::detail::callAtomicOp<THierarchy>(
                addr,
                [&value](auto& ref) { return ref.fetch_and(value); });
        }
    };

    // Or.
    //! The SYCL accelerator atomic operation.
    template<typename T, typename THierarchy>
    struct AtomicOp<AtomicOr, AtomicGenericSycl, T, THierarchy>
    {
        static_assert(std::is_integral_v<T>, "Bitwise operations only supported for integral types.");

        static auto atomicOp(AtomicGenericSycl const&, T* const addr, T const& value) -> T
        {
            return aV2P::detail::callAtomicOp<THierarchy>(addr, [&value](auto& ref) { return ref.fetch_or(value); });
        }
    };

    // Xor.
    //! The SYCL accelerator atomic operation.
    template<typename T, typename THierarchy>
    struct AtomicOp<AtomicXor, AtomicGenericSycl, T, THierarchy>
    {
        static_assert(std::is_integral_v<T>, "Bitwise operations only supported for integral types.");

        static auto atomicOp(AtomicGenericSycl const&, T* const addr, T const& value) -> T
        {
            return aV2P::detail::callAtomicOp<THierarchy>(
                addr,
                [&value](auto& ref) { return ref.fetch_xor(value); });
        }
    };

    // Cas.
    //! The SYCL accelerator atomic operation.
    template<typename T, typename THierarchy>
    struct AtomicOp<AtomicCas, AtomicGenericSycl, T, THierarchy>
    {
        static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>, "SYCL atomics do not support this type");

        static auto atomicOp(AtomicGenericSycl const&, T* const addr, T const& expected, T const& desired) -> T
        {
            auto cas = [&expected, &desired](auto& ref)
            {
                auto expected_ = expected;
                // Atomically compares the value of `ref` with the value of `expected`.
                // If the values are equal, replaces the value of `ref` with `desired`.
                // Otherwise updates `expected` with the value of `ref`.
                // Returns a bool telling us if the exchange happened or not, but the Alpaka API does not make use of
                // it.
                ref.compare_exchange_strong(expected_, desired);

                // If the update succeded, return the previous value of `ref`.
                // Otherwise, return the current value of `ref`.
                return expected_;
            };

            return aV2P::detail::callAtomicOp<THierarchy>(addr, cas);
        }
    };
} // namespace aV2P::trait

#endif

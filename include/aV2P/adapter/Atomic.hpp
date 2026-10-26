/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/adapter/Acc.hpp"
#include "aV2P/atomic/Op.hpp"
#include "aV2P/atomic/Traits.hpp"
#include "aV2P/core/Common.hpp"
#include "aV2P/core/Positioning.hpp"

#include <alpaka/alpaka.hpp>

#include <type_traits>

namespace aV2P::adapter
{
    //! Map the legacy atomic operation tag onto the alpaka 3 operation tag.
    template<typename TOp>
    struct ToAlpakaOperation
    {
        using type = void;
        static constexpr bool mapped = false;
    };

    template<>
    struct ToAlpakaOperation<AtomicAdd>
    {
        using type = alpaka::operation::Add;
        static constexpr bool mapped = true;
    };
    template<>
    struct ToAlpakaOperation<AtomicSub>
    {
        using type = alpaka::operation::Sub;
        static constexpr bool mapped = true;
    };
    template<>
    struct ToAlpakaOperation<AtomicMin>
    {
        using type = alpaka::operation::Min;
        static constexpr bool mapped = true;
    };
    template<>
    struct ToAlpakaOperation<AtomicMax>
    {
        using type = alpaka::operation::Max;
        static constexpr bool mapped = true;
    };
    template<>
    struct ToAlpakaOperation<AtomicExch>
    {
        using type = alpaka::operation::Exch;
        static constexpr bool mapped = true;
    };
    template<>
    struct ToAlpakaOperation<AtomicInc>
    {
        using type = alpaka::operation::Inc;
        static constexpr bool mapped = true;
    };
    template<>
    struct ToAlpakaOperation<AtomicDec>
    {
        using type = alpaka::operation::Dec;
        static constexpr bool mapped = true;
    };
    template<>
    struct ToAlpakaOperation<AtomicAnd>
    {
        using type = alpaka::operation::And;
        static constexpr bool mapped = true;
    };
    template<>
    struct ToAlpakaOperation<AtomicOr>
    {
        using type = alpaka::operation::Or;
        static constexpr bool mapped = true;
    };
    template<>
    struct ToAlpakaOperation<AtomicXor>
    {
        using type = alpaka::operation::Xor;
        static constexpr bool mapped = true;
    };
    template<>
    struct ToAlpakaOperation<AtomicCas>
    {
        using type = alpaka::operation::Cas;
        static constexpr bool mapped = true;
    };

    /** Map the legacy atomic hierarchy onto the alpaka 3 memory scope.
     *
     * `hierarchy::Grids` is "between grids on one device" in legacy alpaka and therefore maps to
     * `scope::Device` (the strongest non-System scope in alpaka 3).  `hierarchy::Blocks` already used
     * device-scope built-ins on the legacy GPU backend and also maps to `scope::Device`.
     * `hierarchy::Threads` used `_block`-scope built-ins and maps to `scope::Block`.
     *
     * alpaka 3 rejects `scope::System` in its atomic entry points with a `static_assert`; the legacy
     * hierarchy has no System level, so no legacy call can reach it.
     */
    template<typename THierarchy>
    constexpr auto toAlpakaScope()
    {
        if constexpr(std::is_same_v<THierarchy, hierarchy::Threads>)
            return alpaka::onAcc::scope::Block{};
        else
            return alpaka::onAcc::scope::Device{};
    }
} // namespace aV2P::adapter

namespace aV2P
{
    namespace trait
    {
        template<
            typename TOp,
            typename T,
            typename THierarchy,
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct AtomicOp<
            TOp,
            AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
            T,
            THierarchy>
        {
            using AlpakaOp = typename adapter::ToAlpakaOperation<TOp>::type;

            AV2P_FN_HOST_ACC static auto atomicOp(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const& acc,
                T* const addr,
                T const& value) -> T
            {
                static_assert(
                    !std::is_void_v<TAccView>,
                    "aV2P: atomic operations require the in-kernel accelerator facade. "
                    "Declare the kernel parameter as a template parameter (`template<typename TAcc> ...`), "
                    "not as a concrete aV2P accelerator type.");
                static_assert(
                    adapter::ToAlpakaOperation<TOp>::mapped,
                    "aV2P: this atomic operation has no alpaka 3 equivalent. Use one of the legacy atomic "
                    "operations Add/Sub/Min/Max/Exch/Inc/Dec/And/Or/Xor/Cas.");
                if constexpr(!std::is_void_v<TAccView> && adapter::ToAlpakaOperation<TOp>::mapped)
                    return alpaka::onAcc::atomicOp<AlpakaOp>(
                        acc.getAlpaka3Acc(),
                        addr,
                        value,
                        adapter::toAlpakaScope<THierarchy>());
                else
                    return T{};
            }

            AV2P_FN_HOST_ACC static auto atomicOp(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const& acc,
                T* const addr,
                T const& compare,
                T const& value) -> T
            {
                static_assert(
                    !std::is_void_v<TAccView>,
                    "aV2P: atomic operations require the in-kernel accelerator facade. "
                    "Declare the kernel parameter as a template parameter (`template<typename TAcc> ...`), "
                    "not as a concrete aV2P accelerator type.");
                static_assert(
                    adapter::ToAlpakaOperation<TOp>::mapped,
                    "aV2P: this atomic operation has no alpaka 3 equivalent. Use one of the legacy atomic "
                    "operations Add/Sub/Min/Max/Exch/Inc/Dec/And/Or/Xor/Cas.");
                if constexpr(!std::is_void_v<TAccView> && adapter::ToAlpakaOperation<TOp>::mapped)
                    return alpaka::onAcc::atomicOp<AlpakaOp>(
                        acc.getAlpaka3Acc(),
                        addr,
                        compare,
                        value,
                        adapter::toAlpakaScope<THierarchy>());
                else
                    return T{};
            }
        };
    } // namespace trait
} // namespace aV2P

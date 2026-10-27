/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/adapter/Acc.hpp"
#include "aV2P/block/sync/Traits.hpp"
#include "aV2P/core/Common.hpp"

#include <alpaka/alpaka.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace aV2P::adapter
{
    //! Linearised index of the current thread within its block.
    template<typename TAcc>
    AV2P_FN_HOST_ACC auto linearBlockThreadIdx(TAcc const& acc) -> std::size_t
    {
        static_assert(sizeof(typename TAcc::Dim) > 0, "aV2P: block primitives need a complete accelerator type.");
        auto const idx = acc.getBlockThreadIdx();
        auto const extent = acc.getWorkDivPayload().blockThreadExtent;
        std::size_t result = 0u;
        for(std::uint32_t i = 0u; i < TAcc::Dim::value; ++i)
            result = result * static_cast<std::size_t>(extent[i]) + static_cast<std::size_t>(idx[i]);
        return result;
    }
} // namespace aV2P::adapter

namespace aV2P
{
    namespace trait
    {
        //! Block barrier on top of `alpaka::onAcc::syncBlockThreads`.
        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct SyncBlockThreads<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>>
        {
            AV2P_FN_ACC static auto syncBlockThreads(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const& acc) -> void
            {
                static_assert(
                    !std::is_void_v<TAccView>,
                    "aV2P: syncBlockThreads requires the in-kernel accelerator facade. "
                    "Declare the kernel parameter as a template parameter (`template<typename TAcc> ...`), "
                    "not as a concrete aV2P accelerator type.");
                if constexpr(!std::is_void_v<TAccView>)
                    alpaka::onAcc::syncBlockThreads(acc.getAlpaka3Acc());
            }
        };

        namespace detail
        {
            //! Shared-memory slot id for the predicate reduction of one combiner type.
            template<typename TOp>
            constexpr std::size_t predicateSharedId()
            {
                if constexpr(std::is_same_v<TOp, BlockCount>)
                    return 0x9e37'79b9'7f4a'7c15ull;
                else if constexpr(std::is_same_v<TOp, BlockAnd>)
                    return 0x9e37'79b9'7f4a'7c16ull;
                else
                    return 0x9e37'79b9'7f4a'7c17ull;
            }

            template<typename TOp>
            inline constexpr bool isPredicateCombiner
                = std::is_same_v<TOp, BlockCount> || std::is_same_v<TOp, BlockAnd> || std::is_same_v<TOp, BlockOr>;
        } // namespace detail

        /** Block-wide predicate reduction portable across host and GPU backends.
         *
         * The algorithm uses one block-shared result cell:
         *  1. thread 0 writes `TOp::InitialValue` into the cell,
         *  2. `syncBlockThreads` makes the initial value visible to the whole block,
         *  3. every thread folds its predicate into the cell with a block-scope atomic operation,
         *  4. `syncBlockThreads` makes all atomic results visible,
         *  5. every thread reads the final value.
         *
         * The three legacy combiners are supported.  `BlockCount` adds one for each non-zero predicate,
         * `BlockAnd` forces the cell to zero if any predicate is zero and `BlockOr` sets it to one if any
         * predicate is non-zero; all three start from the legacy `InitialValue`.  The result therefore
         * equals a sequential fold of `TOp` over the block's predicates, independent of thread order.
         */
        /** Keep the predicate reduction out of line in Intel oneAPI builds.
         *
         * icpx 2026.2 miscompiles this work-group reduction when the function is inlined into a kernel
         * that uses the same shared cell for several reduction rounds: work-items other than the first
         * load a stale value and the block returns the initial value.  A standalone reproducer with
         * `sycl::local_accessor`, `sycl::atomic_ref<..., local_space>` and `nd_item::barrier()` shows the
         * same wrong results (2 0 0 0 / 2 0 0 0 / 2 2 2 2 for three rounds with four work-items);
         * marking the reduction helper `noinline` makes the same program return 2 2 2 2 in every round.
         * The body stays unchanged on other compilers and the attribute costs one call per predicate.
         */
#if defined(AV2P_ACC_SYCL_ENABLED) && defined(__INTEL_LLVM_COMPILER)
#    define AV2P_PREDICATE_NOINLINE __attribute__((noinline))
#else
#    define AV2P_PREDICATE_NOINLINE
#endif

        template<
            typename TOp,
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct SyncBlockThreadsPredicate<TOp, AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>>
        {
            AV2P_PREDICATE_NOINLINE AV2P_FN_ACC static auto syncBlockThreadsPredicate(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const& acc,
                int predicate) -> int
            {
                static_assert(
                    !std::is_void_v<TAccView>,
                    "aV2P: syncBlockThreadsPredicate requires the in-kernel accelerator facade. "
                    "Declare the kernel parameter as a template parameter (`template<typename TAcc> ...`), "
                    "not as a concrete aV2P accelerator type.");

                if constexpr(std::is_void_v<TAccView>)
                {
                    return 0;
                }
                else if constexpr(!detail::isPredicateCombiner<TOp>)
                {
                    static_assert(
                        sizeof(TOp) == 0,
                        "aV2P: syncBlockThreadsPredicate supports the legacy combiners BlockCount, BlockAnd "
                        "and BlockOr. A custom combiner cannot be folded with a portable block-wide atomic "
                        "operation; provide a backend-specific specialization instead.");
                    return 0;
                }
                else
                {
                    auto const& acc3 = acc.getAlpaka3Acc();
                    constexpr std::size_t id = detail::predicateSharedId<TOp>();
                    int& result = alpaka::onAcc::declareSharedVar<int, id>(acc3);

                    if(adapter::linearBlockThreadIdx(acc) == 0u)
                        result = static_cast<int>(TOp::InitialValue);
                    alpaka::onAcc::syncBlockThreads(acc3);

                    if constexpr(std::is_same_v<TOp, BlockCount>)
                    {
                        if(predicate != 0)
                            alpaka::onAcc::atomicAdd(acc3, &result, 1, alpaka::onAcc::scope::Block{});
                    }
                    else if constexpr(std::is_same_v<TOp, BlockAnd>)
                    {
                        if(predicate == 0)
                            alpaka::onAcc::atomicAnd(acc3, &result, 0, alpaka::onAcc::scope::Block{});
                    }
                    else
                    {
                        if(predicate != 0)
                            alpaka::onAcc::atomicOr(acc3, &result, 1, alpaka::onAcc::scope::Block{});
                    }

                    alpaka::onAcc::syncBlockThreads(acc3);
                    return result;
                }
            }
        };
    } // namespace trait
} // namespace aV2P

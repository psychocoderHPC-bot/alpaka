/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/adapter/Acc.hpp"
#include "aV2P/block/shared/dyn/Traits.hpp"
#include "aV2P/block/shared/st/Traits.hpp"
#include "aV2P/core/Common.hpp"

#include <alpaka/alpaka.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace aV2P
{
    namespace trait
    {
        //! Block shared dynamic memory on top of `alpaka::onAcc::getDynSharedMem`.
        template<
            typename T,
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct GetDynSharedMem<T, AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>>
        {
            AV2P_FN_ACC static auto getMem(AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const& acc)
                -> T*
            {
                if constexpr(std::is_void_v<TAccView>)
                {
                    static_assert(
                        !std::is_void_v<TAccView>,
                        "aV2P: block shared memory primitives require the in-kernel accelerator facade. "
                        "Declare the kernel parameter as a template parameter (`template<typename TAcc> ...`), "
                        "not as a concrete aV2P accelerator type.");
                    return nullptr;
                }
                else if constexpr(std::is_same_v<TApi, alpaka::api::Host>)
                {
                    /* Legacy contract: every getDynSharedMem call (independent of T) returns the same
                     * base of the block's dynamic shared-memory region. alpaka 3's host
                     * getDynSharedMem allocates one chunk per call-site id and returns the tail of the
                     * chunk, so the legacy contract is restored explicitly here.
                     *
                     * The dynamic region of a host block shares its storage with the static shared
                     * variables. Allocate (once, keyed by a fixed id) a byte chunk of the full dynamic
                     * size, then derive the chunk base from the returned tail pointer.
                     */
                    constexpr std::size_t id = 0x9e37'79b9'7f4a'7c1dull;
                    auto& acc3 = acc.getAlpaka3Acc();
                    auto const total = static_cast<std::uint32_t>(acc3[alpaka::object::dynSharedMemBytes]);
                    auto* tail = acc3[alpaka::layer::dynShared].template allocDynamic<std::byte, id>(total);
                    return reinterpret_cast<T*>(tail + 1u - total);
                }
                else
                    return alpaka::onAcc::getDynSharedMem<T>(acc.getAlpaka3Acc());
            }
        };

        //! Static block shared variables on top of `alpaka::onAcc::declareSharedVar`.
        template<
            typename T,
            std::size_t TuniqueId,
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct DeclareSharedVar<T, TuniqueId, AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>>
        {
            AV2P_FN_ACC static auto declareVar(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const& acc) -> T&
            {
                static_assert(
                    !std::is_void_v<TAccView>,
                    "aV2P: declareSharedVar requires the in-kernel accelerator facade. "
                    "Declare the kernel parameter as a template parameter (`template<typename TAcc> ...`), "
                    "not as a concrete aV2P accelerator type.");
                if constexpr(!std::is_void_v<TAccView>)
                    return alpaka::onAcc::declareSharedVar<T, TuniqueId>(acc.getAlpaka3Acc());
            }
        };

        //! `freeSharedVars` is a no-op: alpaka 3 shared variables live for one kernel invocation.
        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct FreeSharedVars<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>>
        {
            AV2P_FN_ACC static auto freeVars(AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&)
                -> void
            {
                // alpaka 3 reuses/frees the block shared storage at the end of the kernel invocation.
            }
        };
    } // namespace trait
} // namespace aV2P

/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/adapter/Acc.hpp"
#include "aV2P/core/Common.hpp"
#include "aV2P/mem/fence/Traits.hpp"
#include "aV2P/mem/order/MemoryOrder.hpp"

#include <alpaka/alpaka.hpp>

#include <type_traits>

namespace aV2P::adapter
{
    //! Map the legacy memory order onto the alpaka 3 memory order tag.
    template<typename TMemOrder>
    constexpr auto toAlpakaOrder()
    {
        if constexpr(std::is_same_v<TMemOrder, mem_order::SeqCst>)
            return alpaka::onAcc::order::seqCst;
        else if constexpr(std::is_same_v<TMemOrder, mem_order::AcqRel>)
            return alpaka::onAcc::order::acqRel;
        else if constexpr(std::is_same_v<TMemOrder, mem_order::Release>)
            return alpaka::onAcc::order::release;
        else if constexpr(std::is_same_v<TMemOrder, mem_order::Acquire>)
            return alpaka::onAcc::order::acquire;
        else
            return alpaka::onAcc::order::relaxed;
    }

    /** Map the legacy memory scope onto the alpaka 3 scope.
     *
     * Legacy `memory_scope::Grid` has no alpaka 3 counterpart. The legacy grid scope means "all
     * threads of the current grid", while alpaka 3 models only Block and Device (and System). A grid
     * lives on one device, so `Grid` is conservatively mapped to `scope::Device`, which is at least as
     * strong as the legacy guarantee. `memory_scope::Device` maps to `scope::Device` as well.
     */
    template<typename TMemScope>
    constexpr auto toAlpakaFenceScope()
    {
        if constexpr(std::is_same_v<TMemScope, memory_scope::Block>)
            return alpaka::onAcc::scope::Block{};
        else
            return alpaka::onAcc::scope::Device{};
    }
} // namespace aV2P::adapter

namespace aV2P
{
    namespace trait
    {
        //! Legacy default fence order for the adapter facade (matches the legacy CPU implementations).
        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct MemFenceDefaultOrder<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>>
        {
            // The legacy CUDA/HIP built-in fence defaulted to sequentially consistent; the CPU and
            // SYCL implementations defaulted to acquire-release.
            using type = std::conditional_t<
                std::is_same_v<TApi, alpaka::api::Cuda> || std::is_same_v<TApi, alpaka::api::Hip>,
                mem_order::SeqCst,
                mem_order::AcqRel>;
            static constexpr auto value = []() constexpr
            {
                if constexpr(std::is_same_v<TApi, alpaka::api::Cuda> || std::is_same_v<TApi, alpaka::api::Hip>)
                    return mem_order::seq_cst;
                else
                    return mem_order::acq_rel;
            }();
        };

        template<
            typename TMemOrder,
            typename TMemScope,
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct MemFence<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>, TMemOrder, TMemScope>
        {
            AV2P_FN_ACC static auto mem_fence(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const& acc,
                TMemOrder,
                TMemScope const&) -> void
            {
                static_assert(
                    !std::is_void_v<TAccView>,
                    "aV2P: mem_fence requires the in-kernel accelerator facade. Declare the kernel parameter "
                    "as a template parameter (`template<typename TAcc> ...`), not as a concrete aV2P accelerator "
                    "type.");
                if constexpr(!std::is_void_v<TAccView>)
                    alpaka::onAcc::memFence(
                        acc.getAlpaka3Acc(),
                        adapter::toAlpakaFenceScope<TMemScope>(),
                        adapter::toAlpakaOrder<TMemOrder>());
            }
        };
    } // namespace trait
} // namespace aV2P

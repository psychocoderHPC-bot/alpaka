/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/adapter/Acc.hpp"
#include "aV2P/core/Common.hpp"
#include "aV2P/warp/Traits.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <type_traits>

namespace aV2P::warp
{
    namespace trait
    {
        namespace detail
        {
            template<typename TAccView>
            inline constexpr bool warpHasAccelerator = !std::is_void_v<TAccView>;

        } // namespace detail

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct GetSize<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>>
        {
            AV2P_FN_ACC static auto getSize(AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&)
                -> std::int32_t
            {
                static_assert(
                    detail::warpHasAccelerator<TAccView>,
                    "aV2P: warp primitives require the in-kernel accelerator facade. Declare the kernel parameter as "
                    "a "
                    "template parameter (`template<typename TAcc> ...`), not as a concrete aV2P accelerator type.");
                if constexpr(detail::warpHasAccelerator<TAccView>)
                    return static_cast<std::int32_t>(TAccView::getWarpSize());
                else
                    return 0;
            }
        };

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct GetSizeCompileTime<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>>
        {
            AV2P_FN_ACC static constexpr auto getSizeCompileTime() -> std::int32_t
            {
                if constexpr(detail::warpHasAccelerator<TAccView>)
                    return static_cast<std::int32_t>(TAccView::getWarpSize());
                else
                    return 0;
            }
        };

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct GetSizeUpperLimit<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>>
        {
            AV2P_FN_ACC static constexpr auto getSizeUpperLimit() -> std::int32_t
            {
                if constexpr(detail::warpHasAccelerator<TAccView>)
                    return static_cast<std::int32_t>(TAccView::getWarpSize());
                else
                    return static_cast<std::int32_t>(adapter::warpSizeForTags<TApi, TDeviceKind>);
            }
        };

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct Activemask<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>>
        {
            using Acc = AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>;

            AV2P_FN_ACC static auto activemask(Acc const& acc) -> typename Acc::mask_type
            {
                static_assert(
                    detail::warpHasAccelerator<TAccView>,
                    "aV2P: warp primitives require the in-kernel accelerator facade. Declare the kernel parameter as "
                    "a "
                    "template parameter (`template<typename TAcc> ...`), not as a concrete aV2P accelerator type.");
                if constexpr(detail::warpHasAccelerator<TAccView>)
                    return static_cast<typename Acc::mask_type>(alpaka::onAcc::warp::activemask(acc.getAlpaka3Acc()));
                else
                    return typename Acc::mask_type{};
            }
        };

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct All<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>>
        {
            AV2P_FN_ACC static auto all(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const& acc,
                std::int32_t predicate) -> std::int32_t
            {
                static_assert(
                    detail::warpHasAccelerator<TAccView>,
                    "aV2P: warp primitives require the in-kernel accelerator facade. Declare the kernel parameter as "
                    "a "
                    "template parameter (`template<typename TAcc> ...`), not as a concrete aV2P accelerator type.");
                if constexpr(detail::warpHasAccelerator<TAccView>)
                    return static_cast<std::int32_t>(alpaka::onAcc::warp::all(acc.getAlpaka3Acc(), predicate));
                else
                    return 0;
            }
        };

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct Any<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>>
        {
            AV2P_FN_ACC static auto any(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const& acc,
                std::int32_t predicate) -> std::int32_t
            {
                static_assert(
                    detail::warpHasAccelerator<TAccView>,
                    "aV2P: warp primitives require the in-kernel accelerator facade. Declare the kernel parameter as "
                    "a "
                    "template parameter (`template<typename TAcc> ...`), not as a concrete aV2P accelerator type.");
                if constexpr(detail::warpHasAccelerator<TAccView>)
                    return static_cast<std::int32_t>(alpaka::onAcc::warp::any(acc.getAlpaka3Acc(), predicate));
                else
                    return 0;
            }
        };

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct Ballot<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>>
        {
            using Acc = AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>;

            AV2P_FN_ACC static auto ballot(Acc const& acc, std::int32_t predicate) -> typename Acc::mask_type
            {
                static_assert(
                    detail::warpHasAccelerator<TAccView>,
                    "aV2P: warp primitives require the in-kernel accelerator facade. Declare the kernel parameter as "
                    "a "
                    "template parameter (`template<typename TAcc> ...`), not as a concrete aV2P accelerator type.");
                if constexpr(detail::warpHasAccelerator<TAccView>)
                    return static_cast<typename Acc::mask_type>(
                        alpaka::onAcc::warp::ballot(acc.getAlpaka3Acc(), predicate));
                else
                    return typename Acc::mask_type{};
            }
        };

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct Shfl<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>>
        {
            template<typename T>
            AV2P_FN_ACC static auto shfl(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const& acc,
                T value,
                std::int32_t srcLane,
                std::int32_t width)
            {
                static_assert(
                    detail::warpHasAccelerator<TAccView>,
                    "aV2P: warp primitives require the in-kernel accelerator facade. Declare the kernel parameter as "
                    "a "
                    "template parameter (`template<typename TAcc> ...`), not as a concrete aV2P accelerator type.");
                if constexpr(detail::warpHasAccelerator<TAccView>)
                    return alpaka::onAcc::warp::shfl(
                        acc.getAlpaka3Acc(),
                        value,
                        static_cast<std::uint32_t>(srcLane),
                        static_cast<std::uint32_t>(width));
                else
                    return value;
            }
        };

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct ShflUp<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>>
        {
            template<typename T>
            AV2P_FN_ACC static auto shfl_up(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const& acc,
                T value,
                std::uint32_t offset,
                std::int32_t width)
            {
                static_assert(
                    detail::warpHasAccelerator<TAccView>,
                    "aV2P: warp primitives require the in-kernel accelerator facade. Declare the kernel parameter as "
                    "a "
                    "template parameter (`template<typename TAcc> ...`), not as a concrete aV2P accelerator type.");
                if constexpr(detail::warpHasAccelerator<TAccView>)
                    return alpaka::onAcc::warp::shflUp(
                        acc.getAlpaka3Acc(),
                        value,
                        offset,
                        static_cast<std::uint32_t>(width));
                else
                    return value;
            }
        };

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct ShflDown<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>>
        {
            template<typename T>
            AV2P_FN_ACC static auto shfl_down(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const& acc,
                T value,
                std::uint32_t offset,
                std::int32_t width)
            {
                static_assert(
                    detail::warpHasAccelerator<TAccView>,
                    "aV2P: warp primitives require the in-kernel accelerator facade. Declare the kernel parameter as "
                    "a "
                    "template parameter (`template<typename TAcc> ...`), not as a concrete aV2P accelerator type.");
                if constexpr(detail::warpHasAccelerator<TAccView>)
                    return alpaka::onAcc::warp::shflDown(
                        acc.getAlpaka3Acc(),
                        value,
                        offset,
                        static_cast<std::uint32_t>(width));
                else
                    return value;
            }
        };

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct ShflXor<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>>
        {
            template<typename T>
            AV2P_FN_ACC static auto shfl_xor(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const& acc,
                T value,
                std::int32_t mask,
                std::int32_t width)
            {
                static_assert(
                    detail::warpHasAccelerator<TAccView>,
                    "aV2P: warp primitives require the in-kernel accelerator facade. Declare the kernel parameter as "
                    "a "
                    "template parameter (`template<typename TAcc> ...`), not as a concrete aV2P accelerator type.");
                if constexpr(detail::warpHasAccelerator<TAccView>)
                    return alpaka::onAcc::warp::shflXor(
                        acc.getAlpaka3Acc(),
                        value,
                        static_cast<std::uint32_t>(mask),
                        static_cast<std::uint32_t>(width));
                else
                    return value;
            }
        };
    } // namespace trait
} // namespace aV2P::warp

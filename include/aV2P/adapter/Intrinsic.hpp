/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/adapter/Acc.hpp"
#include "aV2P/core/Common.hpp"
#include "aV2P/intrinsic/Traits.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdint>

namespace aV2P
{
    namespace trait
    {
        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct Popcount<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>>
        {
            template<typename T>
            AV2P_FN_ACC static auto popcount(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&,
                T value) -> std::int32_t
            {
                return alpaka::popcount(value);
            }
        };

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct Ffs<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>>
        {
            template<typename T>
            AV2P_FN_ACC static auto ffs(AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&, T value)
                -> std::int32_t
            {
                return alpaka::ffs(value);
            }
        };
    } // namespace trait
} // namespace aV2P

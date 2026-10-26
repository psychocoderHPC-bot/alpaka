/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/adapter/Acc.hpp"
#include "aV2P/core/Common.hpp"
#include "aV2P/rand/RandDefault.hpp"
#include "aV2P/rand/RandPhilox.hpp"
#include "aV2P/rand/RandStdLib.hpp"
#include "aV2P/rand/Traits.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <type_traits>

namespace aV2P
{
    namespace rand
    {
        namespace engine::trait
        {
            /** Default random engine for the adapter facade.
             *
             * Host accelerators keep the legacy CPU engine (`TinyMersenneTwister`), which is bit-identical
             * to the legacy 2.2 CPU implementation. Device accelerators use the legacy Philox engine;
             * its constructor and state layout are identical to `alpaka::rand::engine::philox::Philox4x32x10`.
             */
            template<
                typename TDim,
                typename TIdx,
                alpaka::concepts::Api TApi,
                alpaka::concepts::DeviceKind TDeviceKind,
                alpaka::concepts::Executor TExecutor,
                typename TAccView>
            struct CreateDefault<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>>
            {
                AV2P_FN_HOST_ACC static auto createDefault(
                    AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&,
                    std::uint32_t const& seed,
                    std::uint32_t const& subsequence,
                    std::uint32_t const& offset)
                {
                    if constexpr(std::is_same_v<TApi, alpaka::api::Host>)
                        return aV2P::rand::engine::cpu::TinyMersenneTwister{seed, subsequence, offset};
                    else
                        return aV2P::rand::Philox4x32x10{seed, subsequence, offset};
                }
            };
        } // namespace engine::trait

        namespace distribution::trait
        {
            template<
                typename T,
                typename TDim,
                typename TIdx,
                alpaka::concepts::Api TApi,
                alpaka::concepts::DeviceKind TDeviceKind,
                alpaka::concepts::Executor TExecutor,
                typename TAccView>
            struct CreateNormalReal<
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
                T,
                std::enable_if_t<std::is_floating_point_v<T>>>
            {
                AV2P_FN_HOST_ACC static auto createNormalReal(
                    AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&) ->
                    alpaka::rand::distribution::NormalReal<T>
                {
                    return {};
                }
            };

            template<
                typename T,
                typename TDim,
                typename TIdx,
                alpaka::concepts::Api TApi,
                alpaka::concepts::DeviceKind TDeviceKind,
                alpaka::concepts::Executor TExecutor,
                typename TAccView>
            struct CreateUniformReal<
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
                T,
                std::enable_if_t<std::is_floating_point_v<T>>>
            {
                AV2P_FN_HOST_ACC static auto createUniformReal(
                    AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&) ->
                    alpaka::rand::distribution::UniformReal<T>
                {
                    // interval::CO == [0, 1), the legacy contract of createUniformReal.
                    return {};
                }
            };

            /** Uniform unsigned integer distribution.
             *
             * alpaka 3 has no integer-result distribution (its `UniformReal` is constrained to floating
             * point). The legacy engine-word based distribution is retained here; it derives the result
             * from the engine's raw words and reproduces the legacy [0, UINT_MAX] contract without an
             * intended bias beyond the legacy implementation.
             */
            template<
                typename T,
                typename TDim,
                typename TIdx,
                alpaka::concepts::Api TApi,
                alpaka::concepts::DeviceKind TDeviceKind,
                alpaka::concepts::Executor TExecutor,
                typename TAccView>
            struct CreateUniformUint<
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
                T,
                std::enable_if_t<std::is_integral_v<T> && std::is_unsigned_v<T>>>
            {
                AV2P_FN_HOST_ACC static auto createUniformUint(
                    AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const&) ->
                    aV2P::rand::distribution::gpu::UniformUint<T>
                {
                    return {};
                }
            };
        } // namespace distribution::trait
    } // namespace rand
} // namespace aV2P

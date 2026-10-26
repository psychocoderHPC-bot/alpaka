/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Common.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <functional>
#include <type_traits>
#include <utility>

namespace aV2P::adapter
{
    /** Concrete alpaka 3 accelerator type used for one legacy launch.
     *
     * The legacy accelerator type is a fixed type (for example `AccCpuSerial<Dim, Idx>`), and some
     * legacy kernels spell that type out instead of deducing the accelerator.  The device primitive
     * adapters therefore have to know the concrete `alpaka::onAcc::Acc<...>` type at the point where the
     * facade is defined.  For the backends implemented here the accelerator type is a pure function of
     * (dimension, index type, api, device kind, executor): the aV2P launch always provides a kernel
     * function with a `dynSharedMemBytes` member and always launches with a `ThreadSpec`, which fixes
     * every optional dictionary entry.
     *
     * The expressions below mirror the accelerator construction in the alpaka 3 backends and only
     * produce a type (`decltype`), they are never evaluated.
     *
     * oneAPI is different: the backend instantiates one kernel (and one accelerator type) per possible
     * sub-group size and selects it at runtime, so there is no single accelerator type per facade.
     * Device primitives are therefore a documented gap for oneAPI (`AlpakaAccType` is `void`).
     */
    template<
        typename TDim,
        typename TIdx,
        typename TApi,
        typename TDeviceKind,
        typename TExecutor>
    struct AlpakaAccType
    {
        using type = void;
    };

    namespace detail
    {
        template<typename TDim, typename TIdx, typename TExecutor>
        struct HostAccType
        {
            using Vec = alpaka::Vec<TIdx, TDim::value>;
            using Spec = alpaka::onHost::ThreadSpec<Vec, Vec, TExecutor>;
            using NumBlocksVec = typename Spec::NumBlocksVecType;
            using NumThreadsVec = typename Spec::NumThreadsVecType;

            static constexpr std::uint32_t simdWidth
                = alpaka::getArchSimdWidth<std::uint8_t>(alpaka::api::host, alpaka::deviceKind::cpu);
            using Shared = alpaka::onAcc::cpu::SingleThreadStaticShared<simdWidth>;

            using type = decltype(alpaka::onAcc::Acc{alpaka::joinDict(
                alpaka::Dict{
                    alpaka::DictEntry(
                        alpaka::layer::block,
                        alpaka::onAcc::cpu::GenericLayer{
                            std::cref(std::declval<NumBlocksVec&>()),
                            std::cref(std::declval<NumBlocksVec&>())}),
                    alpaka::DictEntry(
                        alpaka::layer::thread,
                        alpaka::onAcc::cpu::OneLayer<NumThreadsVec>{}),
                    alpaka::DictEntry(alpaka::layer::shared, std::ref(std::declval<Shared&>())),
                    alpaka::DictEntry(alpaka::action::threadBlockSync, alpaka::onAcc::cpu::NoOp{}),
                    alpaka::DictEntry(
                        alpaka::object::warpSize,
                        std::integral_constant<std::uint32_t, 1u>{})},
                alpaka::Dict{
                    alpaka::DictEntry(alpaka::object::launchedWidthFrameSpec, std::false_type{}),
                    alpaka::DictEntry(alpaka::object::api, alpaka::api::Host{}),
                    alpaka::DictEntry(alpaka::object::deviceKind, alpaka::deviceKind::Cpu{}),
                    alpaka::DictEntry(alpaka::object::exec, TExecutor{}),
                    alpaka::DictEntry(alpaka::layer::dynShared, std::ref(std::declval<Shared&>())),
                    alpaka::DictEntry(
                        alpaka::object::dynSharedMemBytes,
                        std::ref(std::declval<std::uint32_t&>()))})});
        };

#if ALPAKA_LANG_CUDA || ALPAKA_LANG_HIP
        template<typename TDim, typename TIdx, typename TApi, typename TDeviceKind, typename TExecutor>
        struct GpuAccType
        {
            static constexpr std::uint32_t dim = TDim::value;
            using Vec = alpaka::Vec<TIdx, TDim::value>;
            using Spec = alpaka::onHost::ThreadSpec<Vec, Vec, TExecutor>;
            using Optimized = std::conditional_t<
                (dim >= 4u),
                Spec,
                alpaka::onHost::unifiedCudaHip::CallKernel::OptimizedThreadSpec<
                    typename Spec::NumBlocksVecType,
                    typename Spec::NumThreadsVecType>>;
            using BlockLayer = alpaka::onAcc::unifiedCudaHip::BlockLayer<Optimized>;
            using ThreadLayer = alpaka::onAcc::unifiedCudaHip::ThreadLayer<Optimized>;
            using Sync = alpaka::onAcc::unifiedCudaHip::Sync;
            using WarpSize
                = decltype(alpaka::onAcc::unifiedCudaHip::internal::WarpSize::Get<TDeviceKind>{}());

            using type = decltype(alpaka::onAcc::Acc{alpaka::Dict{
                alpaka::DictEntry(
                    alpaka::layer::block,
                    BlockLayer{std::declval<Optimized const&>()}),
                alpaka::DictEntry(
                    alpaka::layer::thread,
                    ThreadLayer{std::declval<Optimized const&>()}),
                alpaka::DictEntry(alpaka::object::launchedWidthFrameSpec, std::false_type{}),
                alpaka::DictEntry(alpaka::action::threadBlockSync, Sync{}),
                alpaka::DictEntry(alpaka::object::api, TApi{}),
                alpaka::DictEntry(alpaka::object::deviceKind, TDeviceKind{}),
                alpaka::DictEntry(alpaka::object::exec, TExecutor{}),
                alpaka::DictEntry(alpaka::object::warpSize, std::declval<WarpSize>())}});
        };
#endif
    } // namespace detail

    template<typename TDim, typename TIdx, typename TExecutor>
    struct AlpakaAccType<TDim, TIdx, alpaka::api::Host, alpaka::deviceKind::Cpu, TExecutor>
    {
        using type = typename detail::HostAccType<TDim, TIdx, TExecutor>::type;
    };

#if ALPAKA_LANG_CUDA
    template<typename TDim, typename TIdx, typename TExecutor>
    struct AlpakaAccType<TDim, TIdx, alpaka::api::Cuda, alpaka::deviceKind::NvidiaGpu, TExecutor>
    {
        using type = typename detail::GpuAccType<
            TDim,
            TIdx,
            alpaka::api::Cuda,
            alpaka::deviceKind::NvidiaGpu,
            TExecutor>::type;
    };
#endif

#if ALPAKA_LANG_HIP
    template<typename TDim, typename TIdx, typename TExecutor>
    struct AlpakaAccType<TDim, TIdx, alpaka::api::Hip, alpaka::deviceKind::AmdGpu, TExecutor>
    {
        using type = typename detail::GpuAccType<
            TDim,
            TIdx,
            alpaka::api::Hip,
            alpaka::deviceKind::AmdGpu,
            TExecutor>::type;
    };
#endif
} // namespace aV2P::adapter

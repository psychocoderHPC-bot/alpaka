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
     * facade is defined.  For CUDA the accelerator type is a pure function of
     * (dimension, index type, api, device kind, executor): the aV2P launch always provides a kernel
     * function with a `dynSharedMemBytes` member and always launches with a `ThreadSpec`, which fixes
     * every optional dictionary entry.
     *
     * The expressions below mirror the accelerator construction in the alpaka 3 backends and only
     * produce a type (`decltype`), they are never evaluated.
     *
     * oneAPI is different: the backend instantiates one kernel (and one accelerator type) per possible
     * sub-group size and selects it at runtime, so no single accelerator type exists per facade at host
     * time.  HIP is different too: the AMD wavefront size is only fixed in device code.  For both the
     * primary template therefore yields `void`, which means "the facade instantiation used at host time
     * does not store the accelerator view".  `adapter::LegacyKernel` re-instantiates the facade class with
     * the concrete in-kernel accelerator type (see `AccAlpaka3::AccView`), so device primitives work for
     * oneAPI and HIP kernels whose accelerator parameter is a template parameter.
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

#if defined(AV2P_ACC_CPU_B_TBB_T_SEQ_ENABLED)
        //! Accelerator type of the alpaka 3 TBB blocks executor.
        //!
        //! `cpu::TbbBlocks` stores the block count in the block layer by value, while `cpu::Serial`
        //! (and therefore `HostAccType`) stores a reference to it.  The dictionary value type is part
        //! of the accelerator type, so the TBB executor needs its own expression to keep the
        //! in-kernel re-instantiation static_assert in `AccAlpaka3` valid.
        template<typename TDim, typename TIdx, typename TExecutor>
        struct TbbHostAccType
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
                            std::cref(std::declval<NumBlocksVec const&>()),
                            std::declval<NumBlocksVec>()}),
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
#endif

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

#if defined(AV2P_ACC_CPU_B_TBB_T_SEQ_ENABLED)
    template<typename TDim, typename TIdx>
    struct AlpakaAccType<TDim, TIdx, alpaka::api::Host, alpaka::deviceKind::Cpu, alpaka::exec::CpuTbbBlocks>
    {
        using type = typename detail::TbbHostAccType<TDim, TIdx, alpaka::exec::CpuTbbBlocks>::type;
    };
#endif

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

// HIP is deliberately not specialized: AMD wavefront size is only known in device code (gfx9 is 64,
// gfx10+ is 32), so a host-computed accelerator type would not match the type the device instantiates
// and the kernel symbol would not resolve.  HIP therefore uses the same in-kernel re-instantiation as
// oneAPI (`TAccView` is `void` at host time); templated accelerator parameters get the concrete
// in-kernel view and device primitives work through it.
} // namespace aV2P::adapter

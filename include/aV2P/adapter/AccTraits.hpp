/* Copyright 2026 Axel Huebl, Benjamin Worpitz, René Widera, Jan Stephan, Bernhard Manfred Gruber, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/adapter/DeviceSpec.hpp"
#include "aV2P/acc/AccDevProps.hpp"
#include "aV2P/acc/Traits.hpp"
#include "aV2P/adapter/Acc.hpp"
#include "aV2P/adapter/Dev.hpp"
#include "aV2P/adapter/Kernel.hpp"
#include "aV2P/adapter/Platform.hpp"
#include "aV2P/adapter/Queue.hpp"
#include "aV2P/adapter/WorkDiv.hpp"
#include "aV2P/core/Common.hpp"
#include "aV2P/idx/Traits.hpp"
#include "aV2P/vec/Traits.hpp"
#include "aV2P/kernel/Traits.hpp"

#include <alpaka/alpaka.hpp>

#include <limits>
#include <string>
#include <type_traits>
#include <utility>

namespace aV2P
{
    namespace trait
    {
        //! Map every implemented legacy tag to its accelerator facade.
        template<typename TDim, typename TIdx>
        struct TagToAcc<aV2P::TagCpuSerial, TDim, TIdx>
        {
            using type = aV2P::AccCpuSerial<TDim, TIdx>;
        };

        template<typename TDim, typename TIdx>
        struct AccToTag<aV2P::AccCpuSerial<TDim, TIdx>>
        {
            using type = aV2P::TagCpuSerial;
        };

#ifdef AV2P_ACC_CPU_B_OMP2_T_SEQ_ENABLED
        template<typename TDim, typename TIdx>
        struct TagToAcc<aV2P::TagCpuOmp2Blocks, TDim, TIdx>
        {
            using type = aV2P::AccCpuOmp2Blocks<TDim, TIdx>;
        };

        template<typename TDim, typename TIdx>
        struct AccToTag<aV2P::AccCpuOmp2Blocks<TDim, TIdx>>
        {
            using type = aV2P::TagCpuOmp2Blocks;
        };
#endif

#ifdef AV2P_ACC_GPU_CUDA_ENABLED
        template<typename TDim, typename TIdx>
        struct TagToAcc<aV2P::TagGpuCudaRt, TDim, TIdx>
        {
            using type = aV2P::AccGpuCudaRt<TDim, TIdx>;
        };

        template<typename TDim, typename TIdx>
        struct AccToTag<aV2P::AccGpuCudaRt<TDim, TIdx>>
        {
            using type = aV2P::TagGpuCudaRt;
        };
#endif

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor>
        struct AccType<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>>
        {
            using type = AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>;
        };

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor>
        struct IsSingleThreadAcc<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>>
            : std::bool_constant<alpaka::exec::isSeqExecutor_v<TExecutor>>
        {
        };

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor>
        struct IsMultiThreadAcc<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>>
            : std::bool_constant<!alpaka::exec::isSeqExecutor_v<TExecutor>>
        {
        };

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor>
        struct GetAccDevProps<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>>
        {
            template<typename TDevSpec>
            AV2P_FN_HOST static auto getAccDevProps(DevAlpaka3<TDevSpec> const& dev) -> AccDevProps<TDim, TIdx>
            {
                auto const prop = dev.getAlpaka3().getDeviceProperties();
                AccDevProps<TDim, TIdx> accProps{
                    // m_multiProcessorCount
                    static_cast<TIdx>(prop.multiProcessorCount),
                    // m_gridBlockExtentMax
                    Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
                    // m_gridBlockCountMax
                    static_cast<TIdx>(std::numeric_limits<TIdx>::max()),
                    // m_blockThreadExtentMax
                    Vec<TDim, TIdx>::all(32u),
                    // m_blockThreadCountMax
                    static_cast<TIdx>(prop.maxThreadsPerBlock),
                    // m_threadElemExtentMax
                    Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
                    // m_threadElemCountMax
                    std::numeric_limits<TIdx>::max(),
                    // m_sharedMemSizeBytes
                    prop.sharedMemPerBlockBytes,
                    // m_globalMemSizeBytes
                    prop.globalMemCapacityBytes,
                    // m_cooperativeLaunch
                    false};
                if constexpr(alpaka::exec::isSeqExecutor_v<TExecutor>)
                {
                    accProps.m_blockThreadCountMax = static_cast<TIdx>(1);
                    accProps.m_blockThreadExtentMax = Vec<TDim, TIdx>::all(1);
                }
                return accProps;
            }
        };

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor>
        struct GetAccName<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>>
        {
            AV2P_FN_HOST static auto getAccName() -> std::string
            {
                return adapter::DeviceSpec<TApi, TDeviceKind>{}.toAlpaka().getName();
            }
        };

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor>
        struct DevType<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>>
        {
            using type = DevAlpaka3<adapter::DeviceSpec<TApi, TDeviceKind>>;
        };

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor>
        struct DimType<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>>
        {
            using type = TDim;
        };

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor>
        struct IdxType<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>>
        {
            using type = TIdx;
        };

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor>
        struct PlatformType<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>>
        {
            using type = PlatformAlpaka3<adapter::DeviceSpec<TApi, TDeviceKind>>;
        };

        template<
            typename TApi,
            typename TDeviceKind,
            typename TExecutor,
            typename TDim,
            typename TIdx,
            typename TDev,
            typename TKernelFn,
            typename... TArgs>
        struct FunctionAttributes<
            AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>,
            TDev,
            TKernelFn,
            TArgs...>
        {
            AV2P_FN_HOST static auto getFunctionAttributes(
                TDev const& dev,
                [[maybe_unused]] TKernelFn const& kernelFn,
                [[maybe_unused]] TArgs&&... args) -> aV2P::KernelFunctionAttributes
            {
                aV2P::KernelFunctionAttributes attributes;
                auto const& props = aV2P::getAccDevProps<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>>(dev);
                attributes.maxThreadsPerBlock = static_cast<int>(props.m_blockThreadCountMax);
                return attributes;
            }
        };

        //! Create a legacy task kernel around an alpaka 3 KernelBundle.
        template<
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TDim,
            typename TIdx,
            typename TWorkDiv,
            typename TKernelFnObj,
            typename... TArgs>
        struct CreateTaskKernel<
            AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>,
            TWorkDiv,
            TKernelFnObj,
            TArgs...>
        {
            AV2P_FN_HOST static auto createTaskKernel(
                TWorkDiv const& workDiv,
                TKernelFnObj const& kernelFnObj,
                TArgs&&... args)
            {
                using Acc = AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>;

                auto const threadElemExtent = aV2P::getWorkDiv<origin::Thread, unit::Elems>(workDiv);
                auto const blockThreadExtent = aV2P::getWorkDiv<origin::Block, unit::Threads>(workDiv);
                auto const dynSharedMemBytes = static_cast<std::uint32_t>(
                    trait::BlockSharedMemDynSizeBytes<TKernelFnObj, Acc>::getBlockSharedMemDynSizeBytes(
                        kernelFnObj,
                        blockThreadExtent,
                        threadElemExtent,
                        args...));

                adapter::LegacyKernel<TKernelFnObj, TDim, TIdx> wrapper{
                    kernelFnObj,
                    threadElemExtent,
                    dynSharedMemBytes};

                return TaskKernelAlpaka3<Acc, TDim, TIdx, decltype(wrapper), TArgs...>(
                    workDiv,
                    wrapper,
                    std::forward<TArgs>(args)...);
            }
        };

        // -----------------------------------------------------------------------------------------
        // Kernel-side bridge: legacy traits on the alpaka 3 accelerator object.
        // -----------------------------------------------------------------------------------------

        template<typename TDict>
        struct DimType<alpaka::onAcc::Acc<TDict>>
        {
            using type = aV2P::DimInt<
                ALPAKA_TYPEOF(std::declval<alpaka::onAcc::Acc<TDict>>()[alpaka::layer::thread].idx())::dim()>;
        };

        template<typename TDict>
        struct IdxType<alpaka::onAcc::Acc<TDict>>
        {
            using type = typename ALPAKA_TYPEOF(
                std::declval<alpaka::onAcc::Acc<TDict>>()[alpaka::layer::thread].idx())::value_type;
        };

        template<typename TDict>
        struct GetWorkDiv<alpaka::onAcc::Acc<TDict>, origin::Grid, unit::Blocks>
        {
            static auto getWorkDiv(alpaka::onAcc::Acc<TDict> const& acc)
            {
                using Dim = typename DimType<ALPAKA_TYPEOF(acc)>::type;
                using Idx = typename IdxType<ALPAKA_TYPEOF(acc)>::type;
                return aV2P::castVec<Idx>(adapter::fromAlpaka(
                    acc.getExtentsOf(alpaka::onAcc::origin::grid, alpaka::onAcc::unit::blocks)))
                    ;
            }
        };

        template<typename TDict>
        struct GetWorkDiv<alpaka::onAcc::Acc<TDict>, origin::Block, unit::Threads>
        {
            static auto getWorkDiv(alpaka::onAcc::Acc<TDict> const& acc)
            {
                using Idx = typename IdxType<ALPAKA_TYPEOF(acc)>::type;
                return aV2P::castVec<Idx>(adapter::fromAlpaka(
                    acc.getExtentsOf(alpaka::onAcc::origin::block, alpaka::onAcc::unit::threads)))
                    ;
            }
        };

        //! The legacy thread-element extent is carried out-of-band, see adapter/WorkDiv.hpp.
        template<typename TDict>
        struct GetWorkDiv<alpaka::onAcc::Acc<TDict>, origin::Thread, unit::Elems>
        {
            static auto getWorkDiv(alpaka::onAcc::Acc<TDict> const& /* acc */)
            {
                using Dim = typename DimType<alpaka::onAcc::Acc<TDict>>::type;
                using Idx = typename IdxType<alpaka::onAcc::Acc<TDict>>::type;
                return adapter::currentThreadElemExtent<Dim, Idx>();
            }
        };

        template<typename TDict>
        struct GetIdx<alpaka::onAcc::Acc<TDict>, origin::Block, unit::Threads>
        {
            template<typename TWorkDiv>
            static auto getIdx(alpaka::onAcc::Acc<TDict> const& acc, TWorkDiv const&)
            {
                using Idx = typename IdxType<ALPAKA_TYPEOF(acc)>::type;
                return aV2P::castVec<Idx>(adapter::fromAlpaka(
                    acc.getIdxWithin(alpaka::onAcc::origin::block, alpaka::onAcc::unit::threads)))
                    ;
            }
        };

        template<typename TDict>
        struct GetIdx<alpaka::onAcc::Acc<TDict>, origin::Grid, unit::Blocks>
        {
            template<typename TWorkDiv>
            static auto getIdx(alpaka::onAcc::Acc<TDict> const& acc, TWorkDiv const&)
            {
                using Idx = typename IdxType<ALPAKA_TYPEOF(acc)>::type;
                return aV2P::castVec<Idx>(adapter::fromAlpaka(
                    acc.getIdxWithin(alpaka::onAcc::origin::grid, alpaka::onAcc::unit::blocks)))
                    ;
            }
        };
    } // namespace trait
} // namespace aV2P

namespace aV2P::interface
{
    //! The alpaka 3 accelerator object implements the legacy accelerator interface.
    template<typename TDict>
    struct ImplementsInterface<InterfaceAcc, alpaka::onAcc::Acc<TDict>>
    {
        static constexpr bool value = true;
    };
} // namespace aV2P::interface

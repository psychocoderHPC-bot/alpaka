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
#include <stdexcept>
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

#ifdef AV2P_ACC_GPU_HIP_ENABLED
        template<typename TDim, typename TIdx>
        struct TagToAcc<aV2P::TagGpuHipRt, TDim, TIdx>
        {
            using type = aV2P::AccGpuHipRt<TDim, TIdx>;
        };

        template<typename TDim, typename TIdx>
        struct AccToTag<aV2P::AccGpuHipRt<TDim, TIdx>>
        {
            using type = aV2P::TagGpuHipRt;
        };
#endif

#ifdef AV2P_ACC_SYCL_ENABLED
        template<typename TDim, typename TIdx>
        struct TagToAcc<aV2P::TagCpuSycl, TDim, TIdx>
        {
            using type = aV2P::AccCpuSycl<TDim, TIdx>;
        };

        template<typename TDim, typename TIdx>
        struct AccToTag<aV2P::AccCpuSycl<TDim, TIdx>>
        {
            using type = aV2P::TagCpuSycl;
        };

        template<typename TDim, typename TIdx>
        struct TagToAcc<aV2P::TagGpuSyclIntel, TDim, TIdx>
        {
            using type = aV2P::AccGpuSyclIntel<TDim, TIdx>;
        };

        template<typename TDim, typename TIdx>
        struct AccToTag<aV2P::AccGpuSyclIntel<TDim, TIdx>>
        {
            using type = aV2P::TagGpuSyclIntel;
        };
#endif

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct AccType<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>>
        {
            using type = AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>;
        };

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct IsSingleThreadAcc<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>>
            : std::bool_constant<alpaka::exec::isSeqExecutor_v<TExecutor>>
        {
        };

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct IsMultiThreadAcc<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>>
            : std::bool_constant<!alpaka::exec::isSeqExecutor_v<TExecutor>>
        {
        };

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct GetAccDevProps<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>>
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
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct GetAccName<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>>
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
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct DevType<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>>
        {
            using type = DevAlpaka3<adapter::DeviceSpec<TApi, TDeviceKind>>;
        };

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct DimType<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>>
        {
            using type = TDim;
        };

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct IdxType<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>>
        {
            using type = TIdx;
        };

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct PlatformType<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>>
        {
            using type = PlatformAlpaka3<adapter::DeviceSpec<TApi, TDeviceKind>>;
        };

        template<
            typename TApi,
            typename TDeviceKind,
            typename TExecutor,
            typename TAccView,
            typename TDim,
            typename TIdx,
            typename TDev,
            typename TKernelFn,
            typename... TArgs>
        struct FunctionAttributes<
            AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
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
                auto const& props = aV2P::getAccDevProps<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>>(dev);
                attributes.maxThreadsPerBlock = static_cast<int>(props.m_blockThreadCountMax);
                if constexpr(adapter::isHostExecutor<TExecutor>)
                {
                    // Legacy CPU backends report the host dynamic shared-memory member allocation here.
                    attributes.maxDynamicSharedSizeBytes
                        = static_cast<int>(aV2P::BlockSharedDynMemberAllocKiB * 1024);
                }
                return attributes;
            }
        };

        //! Create a legacy task kernel around an alpaka 3 KernelBundle.
        template<
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView,
            typename TDim,
            typename TIdx,
            typename TWorkDiv,
            typename TKernelFnObj,
            typename... TArgs>
        struct CreateTaskKernel<
            AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
            TWorkDiv,
            TKernelFnObj,
            TArgs...>
        {
            AV2P_FN_HOST static auto createTaskKernel(
                TWorkDiv const& workDiv,
                TKernelFnObj const& kernelFnObj,
                TArgs&&... args)
            {
                using Acc = AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>;

                auto const payload = adapter::makeWorkDivPayload<TDim, TIdx>(workDiv);

                // Both host executors run one thread per block (CpuOmpBlocks parallelises blocks, not
                // threads).  Keep the legacy 2.2 diagnostic instead of letting the backend throw later.
                if constexpr(alpaka::exec::isSeqExecutor_v<TExecutor>)
                {
                    if(payload.blockThreadExtent.prod() != static_cast<TIdx>(1u))
                    {
                        throw std::runtime_error(
                            "The given work division is not valid for this accelerator: threads per block "
                            "must be 1. Use a different accelerator or a work division with one thread per "
                            "block.");
                    }
                }

                auto const dynSharedMemBytes = static_cast<std::uint32_t>(
                    trait::BlockSharedMemDynSizeBytes<TKernelFnObj, Acc>::getBlockSharedMemDynSizeBytes(
                        kernelFnObj,
                        payload.blockThreadExtent,
                        payload.threadElemExtent,
                        args...));

                adapter::LegacyKernel<Acc, TDim, TIdx, TKernelFnObj> wrapper{kernelFnObj, dynSharedMemBytes};

                return TaskKernelAlpaka3<Acc, TDim, TIdx, decltype(wrapper), TArgs...>(
                    workDiv,
                    wrapper,
                    payload,
                    std::forward<TArgs>(args)...);
            }
        };

        // -----------------------------------------------------------------------------------------
        // Kernel-side bridge: legacy traits on the aV2P acc facade.
        // -----------------------------------------------------------------------------------------
        //
        // The facade is the concrete legacy accelerator type (e.g. `AccCpuSerial<Dim, Idx>`), so kernels
        // with a non-templated accelerator parameter keep compiling.  It stores the current grid-block and
        // block-thread index plus the full `WorkDivPayload`, i.e. only values that survive the device pass.

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct GetWorkDiv<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>, origin::Grid, unit::Blocks>
        {
            AV2P_FN_HOST_ACC static auto getWorkDiv(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const& acc) -> Vec<TDim, TIdx>
            {
                return acc.getWorkDivPayload().gridBlockExtent;
            }
        };

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct GetWorkDiv<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>, origin::Block, unit::Threads>
        {
            AV2P_FN_HOST_ACC static auto getWorkDiv(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const& acc) -> Vec<TDim, TIdx>
            {
                return acc.getWorkDivPayload().blockThreadExtent;
            }
        };

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct GetWorkDiv<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>, origin::Thread, unit::Elems>
        {
            AV2P_FN_HOST_ACC static auto getWorkDiv(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const& acc) -> Vec<TDim, TIdx>
            {
                return acc.getWorkDivPayload().threadElemExtent;
            }
        };

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct GetIdx<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>, origin::Grid, unit::Blocks>
        {
            template<typename TWorkDiv>
            AV2P_FN_HOST_ACC static auto getIdx(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const& acc,
                TWorkDiv const&) -> Vec<TDim, TIdx>
            {
                return acc.getGridBlockIdx() + acc.getWorkDivPayload().gridBlockOffset;
            }
        };

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct GetIdx<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>, origin::Block, unit::Threads>
        {
            template<typename TWorkDiv>
            AV2P_FN_HOST_ACC static auto getIdx(
                AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView> const& acc,
                TWorkDiv const&) -> Vec<TDim, TIdx>
            {
                return acc.getBlockThreadIdx() + acc.getWorkDivPayload().blockThreadOffset;
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

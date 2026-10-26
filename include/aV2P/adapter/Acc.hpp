/* Copyright 2026 Axel Huebl, Benjamin Worpitz, René Widera, Jan Stephan, Bernhard Manfred Gruber, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/acc/Traits.hpp"
#include "aV2P/adapter/DeviceSpec.hpp"
#include "aV2P/adapter/Mapping.hpp"
#include "aV2P/core/Common.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdint>

namespace aV2P
{
    template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    class TaskKernelAlpaka3;

    //! Legacy accelerator facade wrapping an alpaka 3 executor.
    //!
    //! The object itself carries no state.  It is the compile-time identity that maps the legacy
    //! accelerator type onto `alpaka::api`, `alpaka::deviceKind` and `alpaka::exec`.
    template<
        typename TDim,
        typename TIdx,
        alpaka::concepts::Api TApi,
        alpaka::concepts::DeviceKind TDeviceKind,
        alpaka::concepts::Executor TExecutor>
    class AccAlpaka3 final
        : public interface::Implements<InterfaceAcc, AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>>
    {
        static_assert(
            sizeof(TIdx) >= sizeof(int),
            "Index type is not supported, consider using int or a larger type.");

        static_assert(
            aV2P::adapter::isImplementedExecutor<TExecutor>,
            "aV2P Phase 3a implements only AccCpuSerial "
            "(alpaka::api::Host / alpaka::deviceKind::Cpu / alpaka::exec::CpuSerial). "
            "This legacy accelerator is declared in aV2P/adapter/Mapping.hpp, but its executor is "
            "not implemented yet. See Phase 3b.");

    public:
        // Partial specialization with the correct TDim and TIdx is not allowed.
        template<typename TAcc, typename TDim2, typename TIdx2, typename TKernelFnObj, typename... TArgs>
        friend class ::aV2P::TaskKernelAlpaka3;

        using DevSpec = adapter::DeviceSpec<TApi, TDeviceKind>;
        using Executor = TExecutor;
        using Dim = TDim;
        using Idx = TIdx;

        AccAlpaka3(AccAlpaka3 const&) = delete;
        AccAlpaka3(AccAlpaka3&&) = delete;
        auto operator=(AccAlpaka3 const&) -> AccAlpaka3& = delete;
        auto operator=(AccAlpaka3&&) -> AccAlpaka3& = delete;
    };

#ifdef AV2P_ACC_CPU_B_SEQ_T_SEQ_ENABLED
    template<typename TDim, typename TIdx>
    using AccCpuSerial = AccAlpaka3<
        TDim,
        TIdx,
        typename adapter::LegacyAccMapping<TagCpuSerial>::Api,
        typename adapter::LegacyAccMapping<TagCpuSerial>::DeviceKind,
        typename adapter::LegacyAccMapping<TagCpuSerial>::Executor>;
#endif

#ifdef AV2P_ACC_CPU_B_OMP2_T_SEQ_ENABLED
    template<typename TDim, typename TIdx>
    using AccCpuOmp2Blocks = AccAlpaka3<
        TDim,
        TIdx,
        typename adapter::LegacyAccMapping<TagCpuOmp2Blocks>::Api,
        typename adapter::LegacyAccMapping<TagCpuOmp2Blocks>::DeviceKind,
        typename adapter::LegacyAccMapping<TagCpuOmp2Blocks>::Executor>;
#endif

#ifdef AV2P_ACC_GPU_CUDA_ENABLED
    template<typename TDim, typename TIdx>
    using AccGpuCudaRt = AccAlpaka3<
        TDim,
        TIdx,
        typename adapter::LegacyAccMapping<TagGpuCudaRt>::Api,
        typename adapter::LegacyAccMapping<TagGpuCudaRt>::DeviceKind,
        typename adapter::LegacyAccMapping<TagGpuCudaRt>::Executor>;
#endif

#ifdef AV2P_ACC_GPU_HIP_ENABLED
    template<typename TDim, typename TIdx>
    using AccGpuHipRt = AccAlpaka3<
        TDim,
        TIdx,
        typename adapter::LegacyAccMapping<TagGpuHipRt>::Api,
        typename adapter::LegacyAccMapping<TagGpuHipRt>::DeviceKind,
        typename adapter::LegacyAccMapping<TagGpuHipRt>::Executor>;
#endif

#ifdef AV2P_ACC_SYCL_ENABLED
    template<typename TDim, typename TIdx>
    using AccCpuSycl = AccAlpaka3<
        TDim,
        TIdx,
        typename adapter::LegacyAccMapping<TagCpuSycl>::Api,
        typename adapter::LegacyAccMapping<TagCpuSycl>::DeviceKind,
        typename adapter::LegacyAccMapping<TagCpuSycl>::Executor>;

    template<typename TDim, typename TIdx>
    using AccGpuSyclIntel = AccAlpaka3<
        TDim,
        TIdx,
        typename adapter::LegacyAccMapping<TagGpuSyclIntel>::Api,
        typename adapter::LegacyAccMapping<TagGpuSyclIntel>::DeviceKind,
        typename adapter::LegacyAccMapping<TagGpuSyclIntel>::Executor>;
#endif

    // Legacy accelerators without an implementation or without a mapping.  Naming them produces a
    // clear diagnostic instead of a silent, semantically wrong mapping.
    template<typename TDim, typename TIdx, typename TLegacyTag>
    class UnimplementedAcc
    {
        static_assert(
            adapter::isImplementedExecutor<alpaka::exec::CpuSerial> && sizeof(TDim) == 0,
            "aV2P Phase 3a: this legacy accelerator backend is not implemented. "
            "Only AccCpuSerial is available; see Phase 3b.");
    };

#ifdef AV2P_ACC_CPU_B_SEQ_T_THREADS_ENABLED
    template<typename TDim, typename TIdx>
    using AccCpuThreads = UnimplementedAcc<TDim, TIdx, TagCpuThreads>;
#endif

#ifdef AV2P_ACC_CPU_B_SEQ_T_OMP2_ENABLED
    template<typename TDim, typename TIdx>
    using AccCpuOmp2Threads = UnimplementedAcc<TDim, TIdx, TagCpuOmp2Threads>;
#endif

#ifdef AV2P_ACC_CPU_B_TBB_T_SEQ_ENABLED
    template<typename TDim, typename TIdx>
    using AccCpuTbbBlocks = UnimplementedAcc<TDim, TIdx, TagCpuTbbBlocks>;
#endif

#ifdef AV2P_ACC_SYCL_ENABLED
    template<typename TDim, typename TIdx>
    using AccFpgaSyclIntel = UnimplementedAcc<TDim, TIdx, TagFpgaSyclIntel>;

    template<typename TDim, typename TIdx>
    using AccGenericSycl = UnimplementedAcc<TDim, TIdx, TagGenericSycl>;

    template<typename TDim, typename TIdx>
    using AccGpuSyclNvidia = UnimplementedAcc<TDim, TIdx, TagGpuSyclNvidia>;

    template<typename TDim, typename TIdx>
    using AccGpuSyclAmd = UnimplementedAcc<TDim, TIdx, TagGpuSyclAmd>;
#endif
} // namespace aV2P

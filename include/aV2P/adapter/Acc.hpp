/* Copyright 2026 Axel Huebl, Benjamin Worpitz, René Widera, Jan Stephan, Bernhard Manfred Gruber, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/acc/Traits.hpp"
#include "aV2P/adapter/DeviceSpec.hpp"
#include "aV2P/adapter/Mapping.hpp"
#include "aV2P/adapter/WorkDiv.hpp"
#include "aV2P/core/Common.hpp"
#include "aV2P/idx/Traits.hpp"
#include "aV2P/vec/Vec.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <type_traits>

namespace aV2P
{
    template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    class TaskKernelAlpaka3;

    /** Legacy accelerator facade wrapping an alpaka 3 executor.
     *
     * The same concrete type is used in two roles:
     *  - host side it is the compile-time identity that maps the legacy accelerator type onto
     *    `alpaka::api`, `alpaka::deviceKind` and `alpaka::exec`;
     *  - in device code the kernel wrapper constructs it from the raw `alpaka::onAcc::Acc` plus the
     *    trivially-copyable `WorkDivPayload` and passes it to the legacy kernel as `TAcc const&`.
     *
     * The in-kernel instance stores plain values (indices and the work-division payload), never a host
     * pointer or a `thread_local`, so it is valid in CUDA/HIP/SYCL device code and trivially copyable.
     */
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

    public:
        using Payload = adapter::WorkDivPayload<TDim, TIdx>;

        // Partial specialization with the correct TDim and TIdx is not allowed.
        template<typename TAcc, typename TDim2, typename TIdx2, typename TKernelFnObj, typename... TArgs>
        friend class ::aV2P::TaskKernelAlpaka3;

        using DevSpec = adapter::DeviceSpec<TApi, TDeviceKind>;
        using Executor = TExecutor;
        using Dim = TDim;
        using Idx = TIdx;

        /** Host-side tag instance: carries no runtime state.
         *
         * A host-side instance is never passed to a kernel; it only exists because some host-side user code
         * value-initialises an accelerator type.
         */
        AccAlpaka3() = default;

        /** Construct the in-kernel facade from the raw alpaka 3 accelerator and the work-division payload.
         *
         * The raw accelerator is a reference-holding view owned by the backend for the duration of one
         * kernel invocation.  Only values are read from it here.
         */
        template<typename TAlpakaAcc>
        AV2P_FN_ACC AccAlpaka3(TAlpakaAcc const& alpakaAcc, Payload const& payload)
            : m_payload{payload}
            , m_gridBlockIdx{adapter::fromAlpaka(
                  alpakaAcc.getIdxWithin(alpaka::onAcc::origin::grid, alpaka::onAcc::unit::blocks))}
            , m_blockThreadIdx{adapter::fromAlpaka(
                  alpakaAcc.getIdxWithin(alpaka::onAcc::origin::block, alpaka::onAcc::unit::threads))}
        {
        }

        [[nodiscard]] AV2P_FN_HOST_ACC auto getWorkDivPayload() const noexcept -> Payload const&
        {
            return m_payload;
        }

        [[nodiscard]] AV2P_FN_HOST_ACC auto getGridBlockIdx() const noexcept -> Vec<TDim, TIdx> const&
        {
            return m_gridBlockIdx;
        }

        [[nodiscard]] AV2P_FN_HOST_ACC auto getBlockThreadIdx() const noexcept -> Vec<TDim, TIdx> const&
        {
            return m_blockThreadIdx;
        }

    private:
        Payload m_payload{};
        Vec<TDim, TIdx> m_gridBlockIdx{};
        Vec<TDim, TIdx> m_blockThreadIdx{};
    };

    template<
        typename TDim,
        typename TIdx,
        alpaka::concepts::Api TApi,
        alpaka::concepts::DeviceKind TDeviceKind,
        alpaka::concepts::Executor TExecutor>
    inline constexpr bool isTriviallyCopyableAccFacade
        = std::is_trivially_copyable_v<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>>;

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
            sizeof(TDim) == 0,
            "aV2P: this legacy accelerator has no alpaka 3 executor with the same semantics and is not "
            "implemented.  CpuOmp2Threads/CpuThreads/CpuTbbBlocks/FpgaSyclIntel are deliberately not mapped "
            "to an unrelated executor; see aV2P/adapter/Mapping.hpp.");
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

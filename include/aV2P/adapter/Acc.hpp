/* Copyright 2026 Axel Huebl, Benjamin Worpitz, René Widera, Jan Stephan, Bernhard Manfred Gruber, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/acc/Traits.hpp"
#include "aV2P/adapter/AccType.hpp"
#include "aV2P/adapter/DeviceSpec.hpp"
#include "aV2P/adapter/Mapping.hpp"
#include "aV2P/adapter/WorkDiv.hpp"
#include "aV2P/block/shared/dyn/BlockSharedDynMemberAllocKiB.hpp"
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

    namespace adapter
    {
        /** Conservative compile-time warp size derived from the backend tags.
         *
         * The in-kernel facade uses the exact warp size of the underlying alpaka accelerator; this
         * fallback is only needed for facades without a stored accelerator type (oneAPI, see below).
         */
        template<typename TApi, typename TDeviceKind>
        inline constexpr std::uint32_t warpSizeForTags
            = std::is_same_v<TApi, alpaka::api::Host> ? 1u
              : std::is_same_v<TDeviceKind, alpaka::deviceKind::NvidiaGpu> ? 32u
              : std::is_same_v<TDeviceKind, alpaka::deviceKind::AmdGpu>    ? 64u
                                                                           : 32u;
    } // namespace adapter

    /** Legacy accelerator facade wrapping an alpaka 3 executor.
     *
     * The five-parameter public form (see the `AccCpuSerial` / `AccCpuOmp2Blocks` aliases) additionally
     * fixes the concrete `alpaka::onAcc::Acc` type of the current launch (last template parameter).  It
     * is used both as the host-side accelerator identity and as the in-kernel object handed to the user
     * kernel.
     *
     * The in-kernel instance stores the current grid-block and block-thread index plus the full
     * `WorkDivPayload` and a pointer to the backend-owned `alpaka::onAcc::Acc` view.  The view lives for
     * the duration of one kernel invocation, so the pointer never points to host state and never escapes
     * the kernel.  Only plain values are stored in the host-side use of the type.
     */
    template<
        typename TDim,
        typename TIdx,
        alpaka::concepts::Api TApi,
        alpaka::concepts::DeviceKind TDeviceKind,
        alpaka::concepts::Executor TExecutor,
        typename TAccView = typename adapter::AlpakaAccType<TDim, TIdx, TApi, TDeviceKind, TExecutor>::type>
    class AccAlpaka3 final
        : public interface::Implements<
              InterfaceAcc,
              AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>>
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
        using Api = TApi;
        using DeviceKind = TDeviceKind;
        using AccView = TAccView;
        using Dim = TDim;
        using Idx = TIdx;

        static constexpr std::uint32_t warpSize = []() constexpr
        {
            if constexpr(std::is_void_v<TAccView>)
                return adapter::warpSizeForTags<TApi, TDeviceKind>;
            else
                return TAccView::getWarpSize();
        }();
        using mask_type = std::conditional_t<(warpSize <= 32u), std::uint32_t, std::uint64_t>;

        /** Host-side tag instance: carries no runtime state.
         *
         * A host-side instance is never passed to a kernel; it only exists because some host-side user code
         * value-initialises an accelerator type.
         */
        AccAlpaka3() = default;

        /** Construct the in-kernel facade from the raw alpaka 3 accelerator and the work-division payload.
         *
         * The raw accelerator is a reference-holding view owned by the backend for the duration of one
         * kernel invocation.  Only values are read from it here; the pointer to the view is kept so the
         * device primitive adapters can forward to `alpaka::onAcc::*`.
         */
        template<typename TAlpakaAcc>
        AV2P_FN_ACC AccAlpaka3(TAlpakaAcc const& alpakaAcc, Payload const& payload)
            : m_payload{payload}
            , m_gridBlockIdx{adapter::fromAlpaka(
                  alpakaAcc.getIdxWithin(alpaka::onAcc::origin::grid, alpaka::onAcc::unit::blocks))}
            , m_blockThreadIdx{adapter::fromAlpaka(
                  alpakaAcc.getIdxWithin(alpaka::onAcc::origin::block, alpaka::onAcc::unit::threads))}
        {
            if constexpr(!std::is_void_v<TAccView>)
            {
                static_assert(
                    std::is_same_v<std::remove_cvref_t<TAlpakaAcc>, TAccView>,
                    "aV2P: the alpaka accelerator type does not match the type computed for this facade.");
                m_alpaka3Acc = &alpakaAcc;
            }
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

        //! The alpaka 3 accelerator view for the current kernel invocation.
        [[nodiscard]] AV2P_FN_HOST_ACC auto getAlpaka3Acc() const noexcept -> TAccView const&
        {
            static_assert(
                !std::is_void_v<TAccView>,
                "aV2P: this facade has no alpaka 3 accelerator view.  Device primitives are not available "
                "for this backend.");
            return *m_alpaka3Acc;
        }

    private:
        Payload m_payload{};
        Vec<TDim, TIdx> m_gridBlockIdx{};
        Vec<TDim, TIdx> m_blockThreadIdx{};
        TAccView const* m_alpaka3Acc = nullptr;
    };

    template<
        typename TDim,
        typename TIdx,
        alpaka::concepts::Api TApi,
        alpaka::concepts::DeviceKind TDeviceKind,
        alpaka::concepts::Executor TExecutor,
        typename TAccView>
    inline constexpr bool isTriviallyCopyableAccFacade
        = std::is_trivially_copyable_v<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>>;

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

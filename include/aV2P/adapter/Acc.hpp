/* Copyright 2026 Axel Huebl, Benjamin Worpitz, René Widera, Jan Stephan, Bernhard Manfred Gruber, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/acc/Traits.hpp"
#include "aV2P/acc/AccFacadeBase.hpp"
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
     *
     * The oneAPI executor is different: it instantiates one kernel (and therefore one accelerator
     * type) per runtime subgroup size and selects the instantiation at runtime, so no single accelerator
     * type exists at host time.  The public oneAPI facade type carries `void` as `TAccView`, which means
     * "the accelerator view is not stored in this instantiation".  `adapter::LegacyKernel` then passes
     * the kernel functor an instantiation of this same class in which `TAccView` is the concrete
     * `alpaka::onAcc::Acc` type of the current kernel instantiation, and the device primitive adapters
     * forward to `alpaka::onAcc::*` through that view.  For oneAPI the kernel accelerator parameter
     * therefore has to be a template parameter.
     */
    template<
        typename TDim,
        typename TIdx,
        alpaka::concepts::Api TApi,
        alpaka::concepts::DeviceKind TDeviceKind,
        alpaka::concepts::Executor TExecutor,
        typename TAccView = typename adapter::AlpakaAccType<TDim, TIdx, TApi, TDeviceKind, TExecutor>::type>
    class AccAlpaka3
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
        //!
        //! The deduced return type keeps the declaration well-formed for facades without a stored view
        //! (oneAPI public types, where `TAccView` is `void`): the body is only instantiated when the
        //! function is called, which then yields the `static_assert` below instead of an ill-formed
        //! class declaration.
        [[nodiscard]] AV2P_FN_HOST_ACC decltype(auto) getAlpaka3Acc() const noexcept
        {
            static_assert(
                !std::is_void_v<TAccView>,
                "aV2P: this facade instantiation carries no alpaka 3 accelerator view, so device primitives "
                "are not available through it.  For the oneAPI accelerators this is the public accelerator "
                "type: declare the kernel parameter as a template parameter "
                "(`template<typename TAcc> ... void operator()(TAcc const& acc, ...)`) so the in-kernel "
                "facade carries the concrete accelerator type.");
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

    namespace adapter
    {
        //! The `AccAlpaka3` instantiation behind one implemented legacy accelerator tag.
        template<typename TDim, typename TIdx, typename TTag>
        using LegacyAcc = AccAlpaka3<
            TDim,
            TIdx,
            typename LegacyAccMapping<TTag>::Api,
            typename LegacyAccMapping<TTag>::DeviceKind,
            typename LegacyAccMapping<TTag>::Executor>;
    } // namespace adapter

#ifdef AV2P_ACC_CPU_B_SEQ_T_SEQ_ENABLED
    template<typename TDim, typename TIdx>
    class AccCpuSerial : public adapter::PublicAccFacade<adapter::LegacyAcc<TDim, TIdx, TagCpuSerial>>
    {
        using Base = adapter::PublicAccFacade<adapter::LegacyAcc<TDim, TIdx, TagCpuSerial>>;

    public:
        using Base::Base;
    };
#endif

#ifdef AV2P_ACC_CPU_B_OMP2_T_SEQ_ENABLED
    template<typename TDim, typename TIdx>
    class AccCpuOmp2Blocks : public adapter::PublicAccFacade<adapter::LegacyAcc<TDim, TIdx, TagCpuOmp2Blocks>>
    {
        using Base = adapter::PublicAccFacade<adapter::LegacyAcc<TDim, TIdx, TagCpuOmp2Blocks>>;

    public:
        using Base::Base;
    };
#endif

#ifdef AV2P_ACC_GPU_CUDA_ENABLED
    template<typename TDim, typename TIdx>
    class AccGpuCudaRt : public adapter::PublicAccFacade<adapter::LegacyAcc<TDim, TIdx, TagGpuCudaRt>>
    {
        using Base = adapter::PublicAccFacade<adapter::LegacyAcc<TDim, TIdx, TagGpuCudaRt>>;

    public:
        using Base::Base;
    };
#endif

#ifdef AV2P_ACC_GPU_HIP_ENABLED
    template<typename TDim, typename TIdx>
    class AccGpuHipRt : public adapter::PublicAccFacade<adapter::LegacyAcc<TDim, TIdx, TagGpuHipRt>>
    {
        using Base = adapter::PublicAccFacade<adapter::LegacyAcc<TDim, TIdx, TagGpuHipRt>>;

    public:
        using Base::Base;
    };
#endif

// The oneAPI accelerator aliases exist only for the device kinds the build actually targets: the
// public facade selects the alpaka 3 device kind at template time, and a facade whose device kind is
// not compiled into the SYCL kernel set cannot be launched.
#if defined(AV2P_ACC_SYCL_ENABLED) && defined(AV2P_SYCL_TARGET_CPU)
    template<typename TDim, typename TIdx>
    class AccCpuSycl : public adapter::PublicAccFacade<adapter::LegacyAcc<TDim, TIdx, TagCpuSycl>>
    {
        using Base = adapter::PublicAccFacade<adapter::LegacyAcc<TDim, TIdx, TagCpuSycl>>;

    public:
        using Base::Base;
    };
#endif

#if defined(AV2P_ACC_SYCL_ENABLED) && defined(AV2P_SYCL_TARGET_GPU)
    template<typename TDim, typename TIdx>
    class AccGpuSyclIntel : public adapter::PublicAccFacade<adapter::LegacyAcc<TDim, TIdx, TagGpuSyclIntel>>
    {
        using Base = adapter::PublicAccFacade<adapter::LegacyAcc<TDim, TIdx, TagGpuSyclIntel>>;

    public:
        using Base::Base;
    };
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

#define AV2P_DEFINE_UNIMPLEMENTED_ACC(ACC, TAG)                                                                      \
    template<typename TDim, typename TIdx>                                                                           \
    class ACC : public UnimplementedAcc<TDim, TIdx, TAG>                                                            \
    {                                                                                                                \
        using Base = UnimplementedAcc<TDim, TIdx, TAG>;                                                             \
                                                                                                                     \
    public:                                                                                                          \
        using Base::Base;                                                                                            \
    }

#ifdef AV2P_ACC_CPU_B_SEQ_T_THREADS_ENABLED
    AV2P_DEFINE_UNIMPLEMENTED_ACC(AccCpuThreads, TagCpuThreads);
#endif

#ifdef AV2P_ACC_CPU_B_SEQ_T_OMP2_ENABLED
    AV2P_DEFINE_UNIMPLEMENTED_ACC(AccCpuOmp2Threads, TagCpuOmp2Threads);
#endif

#ifdef AV2P_ACC_CPU_B_TBB_T_SEQ_ENABLED
    AV2P_DEFINE_UNIMPLEMENTED_ACC(AccCpuTbbBlocks, TagCpuTbbBlocks);
#endif

#ifdef AV2P_ACC_SYCL_ENABLED
    AV2P_DEFINE_UNIMPLEMENTED_ACC(AccFpgaSyclIntel, TagFpgaSyclIntel);
    AV2P_DEFINE_UNIMPLEMENTED_ACC(AccGenericSycl, TagGenericSycl);
    AV2P_DEFINE_UNIMPLEMENTED_ACC(AccGpuSyclNvidia, TagGpuSyclNvidia);
    AV2P_DEFINE_UNIMPLEMENTED_ACC(AccGpuSyclAmd, TagGpuSyclAmd);
#endif

#undef AV2P_DEFINE_UNIMPLEMENTED_ACC

} // namespace aV2P

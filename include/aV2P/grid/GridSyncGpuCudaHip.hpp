/* Copyright 2026 Mykhailo Varvarin, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/adapter/Kernel.hpp"
#include "aV2P/adapter/WorkDiv.hpp"
#include "aV2P/core/Common.hpp"
#include "aV2P/core/Config.hpp"
#include "aV2P/core/Interface.hpp"
#include "aV2P/grid/Traits.hpp"

#include <cstdint>
#include <type_traits>

#if defined(AV2P_ACC_GPU_CUDA_ENABLED) || defined(AV2P_ACC_GPU_HIP_ENABLED)

namespace aV2P
{
    //! The GPU CUDA/HIP grid synchronization entry point.
    //!
    //! In alpaka 2.2 this was implemented with `cooperative_groups::this_grid().sync()`.  A grid-wide
    //! barrier is only valid in a cooperative kernel launch, and alpaka 3.0.0 does not expose
    //! cooperative launch.  The type and the trait entry point are kept for source compatibility, but
    //! any instantiation that actually asks for the synchronization fails with a diagnostic instead of
    //! silently turning into a no-op.
    class GridSyncCudaHipBuiltIn : public interface::Implements<ConceptGridSync, GridSyncCudaHipBuiltIn>
    {
    };

#    if !defined(AV2P_HOST_ONLY)

#        if defined(AV2P_ACC_GPU_CUDA_ENABLED) && !AV2P_LANG_CUDA
#            error If AV2P_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#        endif

#        if defined(AV2P_ACC_GPU_HIP_ENABLED) && !AV2P_LANG_HIP
#            error If AV2P_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#        endif

    namespace trait
    {
        template<>
        struct SyncGridThreads<GridSyncCudaHipBuiltIn>
        {
            // The diagnostic function is a no-op template.  Do not put
            // AV2P_NO_HOST_ACC_WARNING (`#pragma hd_warning_disable`) in front of it: nvcc requires
            // that pragma to immediately precede a declaration, and the templated friend declaration
            // between the pragma and the function makes it ill-formed.
            template<typename TGridSync = GridSyncCudaHipBuiltIn>
            AV2P_FN_ACC static auto syncGridThreads([[maybe_unused]] TGridSync const& /*gridSync*/) -> void
            {
                static_assert(aV2P::detail::gridSyncUnsupported<TGridSync>, AV2P_GRID_SYNC_COOPERATIVE_MESSAGE);
            }
        };

    } // namespace trait

#        if (defined(AV2P_ACC_GPU_CUDA_ENABLED) && AV2P_LANG_CUDA)                                             \
            || (defined(AV2P_ACC_GPU_HIP_ENABLED) && AV2P_LANG_HIP)

    namespace detail
    {
        /** 2.2 compatibility: the native kernel symbol of one legacy launch.
         *
         * The 2.2 grid-synchronization test queried the occupancy of the native kernel to size the
         * cooperative launch.  The adapter launches through an alpaka 3 `KernelBundle`, so this aliases
         * exactly the `gpuKernel` instantiation that alpaka 3 uses for the same bundle.  It is a
         * compile-only compatibility symbol: aV2P reports `m_cooperativeLaunch == false` on CUDA/HIP and
         * never enters the cooperative-launch path.
         */
        template<typename TKernel, typename TAcc, typename... TArgs>
        inline constexpr auto kernelName = []() constexpr
        {
            using Dim = typename TAcc::Dim;
            using Idx = typename TAcc::Idx;
            using Kernel = adapter::LegacyKernel<TAcc, Dim, Idx, TKernel>;
            using Bundle = alpaka::KernelBundle<Kernel, adapter::WorkDivPayload<Dim, Idx>, TArgs...>;
            using Vec = alpaka::Vec<Idx, static_cast<std::uint32_t>(Dim::value)>;
            using Spec = alpaka::onHost::ThreadSpec<Vec, Vec, typename TAcc::Executor>;
            // aV2P always enqueues a ThreadSpec (never a FrameSpec), which is the `false` template
            // argument of alpaka 3's `gpuKernel`.
            using OptimizedThreadSpec = std::conditional_t<
                (Dim::value >= 4u),
                Spec,
                alpaka::onHost::unifiedCudaHip::CallKernel::OptimizedThreadSpec<
                    typename Spec::NumBlocksVecType,
                    typename Spec::NumThreadsVecType>>;

            return &alpaka::onHost::unifiedCudaHip::gpuKernel<
                typename TAcc::Api,
                typename TAcc::DeviceKind,
                typename TAcc::Executor,
                false,
                Bundle,
                OptimizedThreadSpec>;
        }();
    } // namespace detail

#        endif

#    endif

} // namespace aV2P

#endif

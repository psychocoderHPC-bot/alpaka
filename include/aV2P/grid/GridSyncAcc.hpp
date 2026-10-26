/* Copyright 2026 Mykhailo Varvarin, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/adapter/AccTraits.hpp"
#include "aV2P/grid/Traits.hpp"
#include "aV2P/kernel/Traits.hpp"

#include <alpaka/alpaka.hpp>

#include <type_traits>
#include <utility>

#if defined(_OPENMP)
#    include <omp.h>
#endif

namespace aV2P
{
    namespace trait
    {
        /** Grid synchronization on the alpaka 3 accelerator facade.
         *
         * The facade cannot inherit the 2.2 grid-sync implementation bases (the adapter keeps the
         * accelerator class independent of the legacy backend headers), so the trait is specialized on
         * the facade and selects the implementation from the alpaka 3 executor:
         *
         * - `CpuSerial` executes one block on one thread, so the 2.2 `GridSyncNoOp` is the complete
         *   barrier.
         * - `CpuOmpBlocks` executes one block per OpenMP thread, so the 2.2 OpenMP barrier is the
         *   grid-wide barrier.  A cooperative launch is not needed: the launcher must use as many
         *   blocks as the OpenMP team has threads (see the `MaxActiveBlocks` specialization below).
         * - CUDA, HIP and oneAPI are diagnostic-only, because a grid-wide barrier there needs a
         *   cooperative launch that alpaka 3.0.0 does not provide.
         */
        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView>
        struct SyncGridThreads<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>>
        {
            // No AV2P_NO_HOST_ACC_WARNING here: under nvcc the pragma must immediately precede a
            // declaration, which the templated default argument above prevents.
            template<typename TAcc = AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>>
            AV2P_FN_ACC static auto syncGridThreads([[maybe_unused]] TAcc const& /*acc*/) -> void
            {
                if constexpr(std::is_same_v<TExecutor, alpaka::exec::CpuSerial>)
                {
                    trait::SyncGridThreads<GridSyncNoOp>::syncGridThreads(GridSyncNoOp{});
                }
#if defined(_OPENMP)
                else if constexpr(std::is_same_v<TExecutor, alpaka::exec::CpuOmpBlocks>)
                {
                    trait::SyncGridThreads<GridSyncOmp>::syncGridThreads(GridSyncOmp{});
                }
#endif
                else
                {
                    static_assert(aV2P::detail::gridSyncUnsupported<TAcc>, AV2P_GRID_SYNC_COOPERATIVE_MESSAGE);
                }
            }
        };

        /** Maximum number of blocks that can be active at the same time for a grid-synchronizing kernel.
         *
         * The grid test and the legacy example use this value as the cooperative launch bound.  On the
         * CPU back-ends the whole grid must fit into one launcher wave: one block for the serial
         * executor, and one block per OpenMP team thread for `CpuOmpBlocks`.  The GPU back-ends report
         * zero; their cooperative launch is not available.
         */
        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView,
            typename TDev,
            typename TKernelFnObj,
            typename TWorkDim,
            typename TWorkIdx,
            typename... TArgs>
        struct MaxActiveBlocks<
            AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
            TDev,
            TKernelFnObj,
            TWorkDim,
            TWorkIdx,
            TArgs...>
        {
            AV2P_FN_HOST static auto getMaxActiveBlocks(
                [[maybe_unused]] TKernelFnObj const& /*kernelFnObj*/,
                [[maybe_unused]] TDev const& /*device*/,
                [[maybe_unused]] Vec<TWorkDim, TWorkIdx> const& /*blockThreadExtent*/,
                [[maybe_unused]] Vec<TWorkDim, TWorkIdx> const& /*threadElemExtent*/,
                [[maybe_unused]] TArgs const&... /*args*/) -> int
            {
                if constexpr(std::is_same_v<TExecutor, alpaka::exec::CpuSerial>)
                {
                    // The serial executor runs one block after another on a single thread.
                    return 1;
                }
#if defined(_OPENMP)
                else if constexpr(std::is_same_v<TExecutor, alpaka::exec::CpuOmpBlocks>)
                {
                    // One block per team thread: only then is the OpenMP barrier a grid-wide barrier.
                    return ::omp_get_max_threads();
                }
#endif
                else
                {
                    return 0;
                }
            }
        };

        /** Cooperative kernel task on the accelerator facade.
         *
         * On the host back-ends a cooperative launch needs no special runtime support: the serial
         * executor runs the whole grid on one thread, and `CpuOmpBlocks` keeps one block per OpenMP
         * team thread, so the 2.2 grid barrier semantics hold with the normal launcher.  The GPU
         * back-ends have no cooperative launch in the pinned alpaka 3 and fail with the grid-sync
         * diagnostic instead of pretending the kernel is cooperative.
         */
        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TAccView,
            typename TWorkDiv,
            typename TKernelFnObj,
            typename... TArgs>
        struct CreateTaskCooperativeKernel<
            AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>,
            TWorkDiv,
            TKernelFnObj,
            TArgs...>
        {
            AV2P_FN_HOST static auto createTaskCooperativeKernel(
                TWorkDiv const& workDiv,
                TKernelFnObj const& kernelFnObj,
                TArgs&&... args)
            {
                using Acc = AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor, TAccView>;
                if constexpr(
                    std::is_same_v<TExecutor, alpaka::exec::CpuSerial>
                    || std::is_same_v<TExecutor, alpaka::exec::CpuOmpBlocks>)
                {
                    return trait::CreateTaskKernel<Acc, TWorkDiv, TKernelFnObj, TArgs...>::createTaskKernel(
                        workDiv,
                        kernelFnObj,
                        std::forward<TArgs>(args)...);
                }
                else
                {
                    static_assert(aV2P::detail::gridSyncUnsupported<Acc>, AV2P_GRID_SYNC_COOPERATIVE_MESSAGE);
                }
            }
        };
    } // namespace trait
} // namespace aV2P

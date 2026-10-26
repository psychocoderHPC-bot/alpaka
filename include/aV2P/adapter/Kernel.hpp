/* Copyright 2026 René Widera, Jan Stephan, Bernhard Manfred Gruber, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/adapter/Acc.hpp"
#include "aV2P/adapter/Queue.hpp"
#include "aV2P/adapter/WorkDiv.hpp"
#include "aV2P/core/Common.hpp"
#include "aV2P/kernel/Traits.hpp"
#include "aV2P/workdiv/WorkDivMembers.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <utility>

namespace aV2P::adapter
{
    //! Kernel functor wrapper that carries the legacy per-thread element extent into the kernel.
    template<typename TKernelFnObj, typename TDim, typename TIdx>
    struct LegacyKernel
    {
        TKernelFnObj kernel;
        Vec<TDim, TIdx> threadElemExtent;
        std::uint32_t dynSharedMemBytes = 0u;

        template<typename TAlpakaAcc, typename... TArgs>
        AV2P_FN_ACC void operator()(TAlpakaAcc const& acc, TArgs&&... args) const
        {
            ScopedThreadElemExtent<TDim, TIdx> guard{threadElemExtent};
            kernel(acc, std::forward<TArgs>(args)...);
        }
    };
} // namespace aV2P::adapter

namespace aV2P
{
    //! Legacy kernel execution task bound to an alpaka 3 `KernelBundle`.
    template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    class TaskKernelAlpaka3 : public WorkDivMembers<TDim, TIdx>
    {
    public:
        using Executor = typename TAcc::Executor;
        using KernelBundle = alpaka::KernelBundle<TKernelFnObj, TArgs...>;

        template<typename TWorkDiv>
        AV2P_FN_HOST TaskKernelAlpaka3(
            TWorkDiv const& workDiv,
            TKernelFnObj const& kernelFnObj,
            TArgs&&... args)
            : WorkDivMembers<TDim, TIdx>(workDiv)
            , m_kernelBundle{kernelFnObj, std::forward<TArgs>(args)...}
        {
        }

        [[nodiscard]] AV2P_FN_HOST static constexpr auto getExecutor() -> Executor
        {
            return Executor{};
        }

    public:
        KernelBundle m_kernelBundle;
    };

    namespace trait
    {
        template<
            typename TDev,
            alpaka::concepts::QueueKind TQueueKind,
            typename TAcc,
            typename TDim,
            typename TIdx,
            typename TKernelFnObj,
            typename... TArgs>
        struct Enqueue<QueueAlpaka3<TDev, TQueueKind>, TaskKernelAlpaka3<TAcc, TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            AV2P_FN_HOST static auto enqueue(
                QueueAlpaka3<TDev, TQueueKind>& queue,
                TaskKernelAlpaka3<TAcc, TDim, TIdx, TKernelFnObj, TArgs...> const& task) -> void
            {
                auto const threadSpec = adapter::toThreadSpec(
                    task.m_gridBlockExtent,
                    task.m_blockThreadExtent,
                    task.getExecutor());
                queue.getAlpaka3().enqueue(threadSpec, task.m_kernelBundle);
            }
        };
    } // namespace trait
} // namespace aV2P

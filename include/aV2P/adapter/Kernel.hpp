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
#include <type_traits>
#include <utility>

// The legacy GPU-only kernel-name macros are kept source-compatible.  alpaka 3 derives the kernel name
// from the functor type, so on every backend they expand to nothing.
#ifndef AV2P_KERNEL_NAME
#    define AV2P_KERNEL_NAME(KERNEL, NAME)
#endif
#ifndef AV2P_KERNEL_SCOPED_NAME
#    define AV2P_KERNEL_SCOPED_NAME(KERNEL, NAMESPACE, NAME)
#endif

namespace aV2P::adapter
{
    /** Kernel functor wrapper that builds the legacy acc facade inside the device pass.
     *
     * The legacy work division travels with the launch as the first enqueued kernel argument
     * (`WorkDivPayload`), so it is visible in CUDA/HIP/SYCL device code and per invocation.  No
     * `thread_local` and no host global is involved.  The wrapper additionally carries the legacy static
     * dynamic shared-memory size which alpaka 3 reads from `kernelFn.dynSharedMemBytes`.
     */
    template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj>
    struct LegacyKernel
    {
        static_assert(
            std::is_trivially_copyable_v<TKernelFnObj>,
            "The legacy kernel functor must be trivially copyable to survive the device pass.");
        static_assert(
            isTriviallyCopyableWorkDivPayload<TDim, TIdx>,
            "WorkDivPayload must be trivially copyable to survive the device pass.");

        TKernelFnObj kernel;
        std::uint32_t dynSharedMemBytes = 0u;

        template<typename TAlpakaAcc, typename... TArgs>
        AV2P_FN_ACC void operator()(
            TAlpakaAcc const& alpakaAcc,
            WorkDivPayload<TDim, TIdx> const& payload,
            TArgs const&... args) const
        {
            TAcc const legacyAcc{alpakaAcc, payload};
            kernel(legacyAcc, args...);
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
        using Payload = adapter::WorkDivPayload<TDim, TIdx>;
        using KernelBundle = alpaka::KernelBundle<TKernelFnObj, Payload, TArgs...>;

        template<typename TWorkDiv>
        AV2P_FN_HOST TaskKernelAlpaka3(
            TWorkDiv const& workDiv,
            TKernelFnObj const& kernelFnObj,
            Payload const& payload,
            TArgs&&... args)
            : WorkDivMembers<TDim, TIdx>(workDiv)
            , m_kernelBundle{kernelFnObj, payload, std::forward<TArgs>(args)...}
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

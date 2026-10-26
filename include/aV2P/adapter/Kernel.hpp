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
    namespace detail
    {
        //! Dependent false: keeps the unreachable diagnostic below dependent so ahead-of-time device
        //! front ends (nvcc/EDG) do not evaluate it while discarding the oneAPI branch.
        template<typename>
        inline constexpr bool dependentFalse = false;
    } // namespace detail

    /** Kernel functor wrapper that builds the legacy acc facade inside the device pass.
     *
     * The legacy work division travels with the launch as the first enqueued kernel argument
     * (`WorkDivPayload`), so it is visible in CUDA/HIP/SYCL device code and per invocation.  No
     * `thread_local` and no host global is involved.  The wrapper additionally carries the legacy static
     * dynamic shared-memory size which alpaka 3 reads from `kernelFn.dynSharedMemBytes`.
     *
     * For the Host/CUDA/HIP executors the alpaka accelerator is a pure function of the legacy launch
     * shape, so the public facade type (`TAcc`, which stores the accelerator view) is constructed
     * directly.  The oneAPI executor instantiates one kernel (and one accelerator type) per runtime
     * subgroup size, so the concrete accelerator type is only known here, in the kernel.  For oneAPI
     * the wrapper therefore instantiates the same `AccAlpaka3` class with the actual
     * `alpaka::onAcc::Acc<...>` type (see `AccAlpaka3::AccView`) and hands that runtime facade to the
     * kernel functor.  Kernels written against a single public oneAPI accelerator type (non-templated
     * accelerator parameter) keep the 2.2 call convention and receive the payload-only public facade,
     * whose device primitives remain unavailable for the same reason.
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

        //! The facade the user kernel calls: the public facade type where the alpaka accelerator type is
        //! known at host time, and the same class template instantiated with the concrete in-kernel
        //! accelerator type otherwise (oneAPI).
        template<typename TAlpakaAcc>
        using FacadeForAlpakaAcc = AccAlpaka3<
            TDim,
            TIdx,
            typename TAcc::Api,
            typename TAcc::DeviceKind,
            typename TAcc::Executor,
            std::remove_cvref_t<TAlpakaAcc>>;

        template<typename TAlpakaAcc, typename... TArgs>
        AV2P_FN_ACC void operator()(
            TAlpakaAcc const& alpakaAcc,
            WorkDivPayload<TDim, TIdx> const& payload,
            TArgs const&... args) const
        {
            if constexpr(!std::is_void_v<typename TAcc::AccView>)
            {
                // Host/CUDA/HIP: the accelerator type is fixed by the launch shape.
                TAcc const legacyAcc{alpakaAcc, payload};
                kernel(legacyAcc, args...);
            }
            else
            {
                // oneAPI: one kernel instantiation per runtime subgroup size.
                if constexpr(std::is_invocable_v<TKernelFnObj const&, FacadeForAlpakaAcc<TAlpakaAcc>, TArgs const&...>)
                {
                    // Generic accelerator parameter (a function template, or a functor whose class
                    // template parameter is the accelerator): pass the in-kernel instantiation of the
                    // facade that carries the concrete alpaka 3 accelerator type, so device primitives
                    // forward to `alpaka::onAcc::*`.
                    FacadeForAlpakaAcc<TAlpakaAcc> const legacyAcc{alpakaAcc, payload};
                    kernel(legacyAcc, args...);
                }
                else if constexpr(std::is_invocable_v<TKernelFnObj const&, TAcc, TArgs const&...>)
                {
                    // 2.2 call convention: the functor takes the public accelerator type and nothing
                    // else, so it receives the payload-only public facade.  Device primitives are not
                    // available through it, because no single alpaka 3 accelerator type exists at
                    // host time for this backend.
                    TAcc const legacyAcc{alpakaAcc, payload};
                    kernel(legacyAcc, args...);
                }
                else
                {
                    static_assert(
                        detail::dependentFalse<TKernelFnObj>,
                        "aV2P: this kernel functor cannot accept the accelerator facade of the current "
                        "oneAPI kernel instantiation or the public accelerator type.  Declare the kernel "
                        "parameter as a template parameter "
                        "(`template<typename TAcc> ... void operator()(TAcc const& acc, ...)`) instead of "
                        "a concrete aV2P accelerator type to get access to the device primitives.");
                }
            }
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

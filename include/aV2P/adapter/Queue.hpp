/* Copyright 2026 Benjamin Worpitz, Matthias Werner, Jan Stephan, Bernhard Manfred Gruber, Jeffrey Kelling
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/adapter/Dev.hpp"
#include "aV2P/core/Common.hpp"
#include "aV2P/dev/Traits.hpp"
#include "aV2P/event/Traits.hpp"
#include "aV2P/queue/Traits.hpp"
#include "aV2P/wait/Traits.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/core/CallbackThread.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <type_traits>

namespace aV2P
{
    template<typename TDev>
    class EventAlpaka3;

    namespace detail
    {
        //! True only for the CUDA/HIP adapter devices, whose queue native handle is the runtime stream.
        //!
        //! The 2.2 CUDA/HIP queues exposed a stream and could be constructed from one.  alpaka 3
        //! `onHost::Queue` always creates and owns its stream, so the adapter keeps a separate
        //! non-owning stream-backed mode for that compatibility path.
        template<typename TDev>
        inline constexpr bool isUniformCudaHipDevice = false;

#if defined(AV2P_ACC_GPU_CUDA_ENABLED)
        template<>
        inline constexpr bool isUniformCudaHipDevice<DevCudaRt> = true;
#endif
#if defined(AV2P_ACC_GPU_HIP_ENABLED)
        template<>
        inline constexpr bool isUniformCudaHipDevice<DevHipRt> = true;
#endif
    } // namespace detail

    //! Queue handle wrapping an alpaka 3 `onHost::Queue`.
    //!
    //! `TQueueKind` is the alpaka 3 queue kind (`alpaka::queueKind::Blocking` or
    //! `alpaka::queueKind::NonBlocking`).  The queue kind is honoured at construction time.
    //! `getAlpaka3()` returns a reference into the wrapper and is bound to the wrapper's lifetime.
    //!
    //! On CUDA/HIP the wrapper can additionally adopt an existing runtime stream, mirroring the 2.2
    //! `QueueUniformCudaHipRt` constructor.  Such a wrapper does not own the stream: it never
    //! synchronizes or destroys it and `getAlpaka3()` is unavailable because alpaka 3 has no queue
    //! that borrows a native stream.  `getNativeHandle()` and `getDev()` remain valid, and wait/empty
    //! and host-task enqueue operate directly on the borrowed stream.
    template<typename TDev, alpaka::concepts::QueueKind TQueueKind>
    class QueueAlpaka3 final
        : public interface::Implements<ConceptCurrentThreadWaitFor, QueueAlpaka3<TDev, TQueueKind>>
        , public interface::Implements<ConceptQueue, QueueAlpaka3<TDev, TQueueKind>>
        , public interface::Implements<ConceptGetDev, QueueAlpaka3<TDev, TQueueKind>>
    {
    public:
        using QueueType
            = alpaka::onHost::Queue<typename TDev::DevType, alpaka::onHost::QueuePolicyList<TQueueKind>>;
        using NativeHandleType = decltype(std::declval<QueueType const&>().getNativeHandle());

        explicit QueueAlpaka3(TDev const& dev)
            : m_dev{dev}
            , m_queueImpl{createQueue(dev)}
        {
        }

#if defined(AV2P_ACC_GPU_CUDA_ENABLED) || defined(AV2P_ACC_GPU_HIP_ENABLED)
        //! Adopt an existing CUDA/HIP stream without creating a queue.
        //!
        //! The wrapper is non-owning: the caller keeps responsibility for the stream lifetime.  The
        //! device is the one that is current when the wrapper is constructed, matching 2.2.
        template<typename TStream>
        requires(detail::isUniformCudaHipDevice<TDev> && std::same_as<std::decay_t<TStream>, NativeHandleType>)
        explicit QueueAlpaka3(TStream stream)
            : m_dev{makeCurrentDevice()}
            , m_nativeStream{stream}
            , m_callbackThread{std::make_shared<alpaka::core::CallbackThread>()}
        {
        }
#endif

        //! Non-owning reference to the wrapped alpaka 3 queue.
        //!
        //! Only available for queues that own an alpaka 3 queue.  A stream-adopting wrapper throws
        //! `std::bad_optional_access` (terminating, because the accessor is `noexcept`) instead of
        //! silently creating a second queue.
        [[nodiscard]] auto getAlpaka3() & noexcept -> QueueType&
        {
            return m_queueImpl.value();
        }

        [[nodiscard]] auto getAlpaka3() const& noexcept -> QueueType const&
        {
            return m_queueImpl.value();
        }

        //! The native stream for CUDA/HIP, the host queue id for CPU and the backend handle otherwise.
        [[nodiscard]] auto getNativeHandle() const noexcept
        {
            if(m_nativeStream)
                return *m_nativeStream;
            return m_queueImpl->getNativeHandle();
        }

        [[nodiscard]] auto getDev() const noexcept -> TDev const&
        {
            return m_dev;
        }

        [[nodiscard]] bool isNativeStreamBacked() const noexcept
        {
            return m_nativeStream.has_value();
        }

        //! Wait for all work in the queue, using the alpaka 3 queue or the borrowed stream.
        void wait() const
        {
            if(m_queueImpl)
                alpaka::onHost::wait(*m_queueImpl);
            else
                nativeWait();
        }

        [[nodiscard]] bool empty() const
        {
            if(m_queueImpl)
                return m_queueImpl->isEmpty();
            return nativeEmpty();
        }

        //! Enqueue a host task, preserving the blocking/non-blocking queue kind on both paths.
        template<typename TTask>
        void enqueueHostFn(TTask const& task) const
        {
            if(m_queueImpl)
            {
                m_queueImpl->enqueueHostFn(task);
                return;
            }

            if constexpr(detail::isUniformCudaHipDevice<TDev>)
            {
                struct TaskData
                {
                    std::shared_ptr<alpaka::core::CallbackThread> callbackThread;
                    TTask task;
                };

                auto* data = new TaskData{m_callbackThread, task};
                auto const hostFunc = [](void* raw) noexcept
                {
                    std::unique_ptr<TaskData> holder{static_cast<TaskData*>(raw)};
                    auto callbackThread = holder->callbackThread;
                    auto localTask = std::move(holder->task);
                    holder.reset();
                    callbackThread->submit([task = std::move(localTask)]() mutable { task(); }).wait();
                };

#    if defined(AV2P_ACC_GPU_CUDA_ENABLED)
                if constexpr(std::is_same_v<TDev, DevCudaRt>)
                    cudaLaunchHostFunc(*m_nativeStream, hostFunc, data);
#    endif
#    if defined(AV2P_ACC_GPU_HIP_ENABLED)
                if constexpr(std::is_same_v<TDev, DevHipRt>)
                    hipLaunchHostFunc(*m_nativeStream, hostFunc, data);
#    endif

                if constexpr(std::is_same_v<TQueueKind, alpaka::queueKind::Blocking>)
                    nativeWait();
            }
        }

        auto operator==(QueueAlpaka3 const& rhs) const -> bool
        {
            if(m_queueImpl && rhs.m_queueImpl)
                return *m_queueImpl == *rhs.m_queueImpl;
            return m_nativeStream == rhs.m_nativeStream && m_dev == rhs.m_dev;
        }

        auto operator!=(QueueAlpaka3 const& rhs) const -> bool
        {
            return !(*this == rhs);
        }

    private:
        //! `alpaka::onHost::Device::makeQueue` is not const, so take the legacy device handle by value.
        //! The device copy shares the underlying handle and keeps the created queue's device alive.
        static auto createQueue(TDev dev) -> QueueType
        {
            return dev.getAlpaka3().makeQueue(TQueueKind{});
        }

        static auto makeCurrentDevice() -> TDev
        {
            int deviceIdx = 0;
#if defined(AV2P_ACC_GPU_CUDA_ENABLED)
            if constexpr(std::is_same_v<TDev, DevCudaRt>)
                cudaGetDevice(&deviceIdx);
#endif
#if defined(AV2P_ACC_GPU_HIP_ENABLED)
            if constexpr(std::is_same_v<TDev, DevHipRt>)
                hipGetDevice(&deviceIdx);
#endif
            using DevSpec = typename TDev::DevSpec;
            return TDev{alpaka::onHost::makeDeviceSelector(DevSpec{}.toAlpaka())
                            .makeDevice(static_cast<std::uint32_t>(deviceIdx))};
        }

        void nativeWait() const
        {
            if constexpr(detail::isUniformCudaHipDevice<TDev>)
            {
#if defined(AV2P_ACC_GPU_CUDA_ENABLED)
                if constexpr(std::is_same_v<TDev, DevCudaRt>)
                    cudaStreamSynchronize(*m_nativeStream);
#endif
#if defined(AV2P_ACC_GPU_HIP_ENABLED)
                if constexpr(std::is_same_v<TDev, DevHipRt>)
                    hipStreamSynchronize(*m_nativeStream);
#endif
            }
        }

        [[nodiscard]] bool nativeEmpty() const
        {
            if constexpr(detail::isUniformCudaHipDevice<TDev>)
            {
#if defined(AV2P_ACC_GPU_CUDA_ENABLED)
                if constexpr(std::is_same_v<TDev, DevCudaRt>)
                    return cudaStreamQuery(*m_nativeStream) == cudaSuccess;
#endif
#if defined(AV2P_ACC_GPU_HIP_ENABLED)
                if constexpr(std::is_same_v<TDev, DevHipRt>)
                    return hipStreamQuery(*m_nativeStream) == hipSuccess;
#endif
            }
            return true;
        }

        TDev m_dev;
        //! Engaged for device-constructed, alpaka 3 owning queues.  Empty for stream-adopting wrappers.
        std::optional<QueueType> m_queueImpl;
        //! Borrowed CUDA/HIP stream; engaged only for stream-adopting wrappers.
        std::optional<NativeHandleType> m_nativeStream;
        //! Executes host tasks launched into the borrowed stream.  Only set for stream-adopting wrappers.
        std::shared_ptr<alpaka::core::CallbackThread> m_callbackThread;
    };

    using QueueCpuBlocking = QueueAlpaka3<DevCpu, alpaka::queueKind::Blocking>;
    using QueueCpuNonBlocking = QueueAlpaka3<DevCpu, alpaka::queueKind::NonBlocking>;

#ifdef AV2P_ACC_GPU_CUDA_ENABLED
    using QueueCudaRtBlocking = QueueAlpaka3<DevCudaRt, alpaka::queueKind::Blocking>;
    using QueueCudaRtNonBlocking = QueueAlpaka3<DevCudaRt, alpaka::queueKind::NonBlocking>;
#endif

#ifdef AV2P_ACC_GPU_HIP_ENABLED
    using QueueHipRtBlocking = QueueAlpaka3<DevHipRt, alpaka::queueKind::Blocking>;
    using QueueHipRtNonBlocking = QueueAlpaka3<DevHipRt, alpaka::queueKind::NonBlocking>;
#endif

#ifdef AV2P_ACC_SYCL_ENABLED
    using QueueCpuSyclBlocking = QueueAlpaka3<DevCpuSycl, alpaka::queueKind::Blocking>;
    using QueueCpuSyclNonBlocking = QueueAlpaka3<DevCpuSycl, alpaka::queueKind::NonBlocking>;
    using QueueGpuSyclIntelBlocking = QueueAlpaka3<DevGpuSyclIntel, alpaka::queueKind::Blocking>;
    using QueueGpuSyclIntelNonBlocking = QueueAlpaka3<DevGpuSyclIntel, alpaka::queueKind::NonBlocking>;

    //! 2.2 queue name: the queue of the oneAPI tag `TTag`.
    template<typename TTag>
    using QueueGenericSyclBlocking = QueueAlpaka3<DevGenericSycl<TTag>, alpaka::queueKind::Blocking>;

    template<typename TTag>
    using QueueGenericSyclNonBlocking = QueueAlpaka3<DevGenericSycl<TTag>, alpaka::queueKind::NonBlocking>;

    using QueueGpuSyclAmdBlocking = QueueGenericSyclBlocking<TagGpuSyclAmd>;
    using QueueGpuSyclAmdNonBlocking = QueueGenericSyclNonBlocking<TagGpuSyclAmd>;
    using QueueGpuSyclNvidiaBlocking = QueueGenericSyclBlocking<TagGpuSyclNvidia>;
    using QueueGpuSyclNvidiaNonBlocking = QueueGenericSyclNonBlocking<TagGpuSyclNvidia>;
#endif

    //! 2.2 generic-threads queue names: the queue of an arbitrary legacy device.
    template<typename TDev>
    using QueueGenericThreadsBlocking = QueueAlpaka3<TDev, alpaka::queueKind::Blocking>;

    template<typename TDev>
    using QueueGenericThreadsNonBlocking = QueueAlpaka3<TDev, alpaka::queueKind::NonBlocking>;

    //! 2.2 uniform CUDA/HIP queue names: the queue of the device selected by the runtime API tag.
    template<typename TApi>
    using QueueUniformCudaHipRtBlocking = QueueAlpaka3<DevUniformCudaHipRt<TApi>, alpaka::queueKind::Blocking>;

    template<typename TApi>
    using QueueUniformCudaHipRtNonBlocking
        = QueueAlpaka3<DevUniformCudaHipRt<TApi>, alpaka::queueKind::NonBlocking>;

    namespace trait
    {
        template<typename TDev, alpaka::concepts::QueueKind TQueueKind>
        struct DevType<QueueAlpaka3<TDev, TQueueKind>>
        {
            using type = TDev;
        };

        template<typename TDev, alpaka::concepts::QueueKind TQueueKind>
        struct GetDev<QueueAlpaka3<TDev, TQueueKind>>
        {
            AV2P_FN_HOST static auto getDev(QueueAlpaka3<TDev, TQueueKind> const& queue) -> TDev
            {
                return queue.getDev();
            }
        };

        //! The 2.2 native queue handle trait: the CUDA/HIP stream, the host queue id, or the backend handle.
        template<typename TDev, alpaka::concepts::QueueKind TQueueKind>
        struct NativeHandle<QueueAlpaka3<TDev, TQueueKind>>
        {
            [[nodiscard]] static auto getNativeHandle(QueueAlpaka3<TDev, TQueueKind> const& queue)
            {
                return queue.getNativeHandle();
            }
        };

        //! Generic fallback: enqueue the legacy task as a host function on the wrapped alpaka 3
        //! queue.  Kernel and memory tasks have more specific specializations.
        template<typename TDev, typename TTask, alpaka::concepts::QueueKind TQueueKind>
        struct Enqueue<QueueAlpaka3<TDev, TQueueKind>, TTask>
        {
            AV2P_FN_HOST static auto enqueue(QueueAlpaka3<TDev, TQueueKind>& queue, TTask const& task) -> void
            {
                queue.enqueueHostFn(task);
            }
        };

        template<typename TDev, alpaka::concepts::QueueKind TQueueKind>
        struct Empty<QueueAlpaka3<TDev, TQueueKind>>
        {
            AV2P_FN_HOST static auto empty(QueueAlpaka3<TDev, TQueueKind> const& queue) -> bool
            {
                return queue.empty();
            }
        };

        template<typename TDev, alpaka::concepts::QueueKind TQueueKind>
        struct CurrentThreadWaitFor<QueueAlpaka3<TDev, TQueueKind>>
        {
            AV2P_FN_HOST static auto currentThreadWaitFor(QueueAlpaka3<TDev, TQueueKind> const& queue) -> void
            {
                queue.wait();
            }
        };
    } // namespace trait
} // namespace aV2P

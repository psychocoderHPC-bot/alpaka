/* Copyright 2025 Benjamin Worpitz, Matthias Werner, René Widera, Andrea Bocci, Bernhard Manfred Gruber,
 * Antonio Di Pilato
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/CallbackThread.hpp"
#include "aV2P/core/Cuda.hpp"
#include "aV2P/core/Hip.hpp"
#include "aV2P/core/Interface.hpp"
#include "aV2P/dev/Traits.hpp"
#include "aV2P/event/Traits.hpp"
#include "aV2P/meta/DependentFalseType.hpp"
#include "aV2P/platform/Traits.hpp"
#include "aV2P/queue/Traits.hpp"
#include "aV2P/traits/Traits.hpp"
#include "aV2P/wait/Traits.hpp"

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <thread>

#if defined(AV2P_ACC_GPU_CUDA_ENABLED) || defined(AV2P_ACC_GPU_HIP_ENABLED)

namespace aV2P
{
    template<typename TApi>
    class EventUniformCudaHipRt;

    template<typename TApi>
    class DevUniformCudaHipRt;

    template<typename TApi>
    struct PlatformUniformCudaHipRt;

    namespace uniform_cuda_hip::detail
    {
        //! The CUDA/HIP RT queue implementation.
        template<typename TApi>
        class QueueUniformCudaHipRtImpl final
        {
        public:
            AV2P_FN_HOST QueueUniformCudaHipRtImpl(DevUniformCudaHipRt<TApi> const& dev)
                : m_dev(dev)
                , m_UniformCudaHipQueue()
                , m_isOwning(true)
            {
                AV2P_DEBUG_MINIMAL_LOG_SCOPE;

                // Set the current device.
                AV2P_UNIFORM_CUDA_HIP_RT_CHECK(TApi::setDevice(m_dev.getNativeHandle()));

                // - [cuda/hip]StreamDefault: Default queue creation flag.
                // - [cuda/hip]StreamNonBlocking: Specifies that work running in the created queue may run
                // concurrently with work in queue 0 (the NULL queue),
                // and that the created queue should perform no implicit synchronization with queue 0.
                // Create the queue on the current device.
                // NOTE: [cuda/hip]StreamNonBlocking is required to match the semantic implemented in the alpaka
                // CPU queue. It would be too much work to implement implicit default queue synchronization on CPU.

                AV2P_UNIFORM_CUDA_HIP_RT_CHECK(
                    TApi::streamCreateWithFlags(&m_UniformCudaHipQueue, TApi::streamNonBlocking));
            }

            AV2P_FN_HOST QueueUniformCudaHipRtImpl(typename TApi::Stream_t stream)
                : m_dev(aV2P::getDevByIdx(
                    aV2P::PlatformUniformCudaHipRt<TApi>{},
                    static_cast<std::size_t>(TApi::getCurrentDevice())))
                , m_UniformCudaHipQueue(stream)
                , m_isOwning(false)
            {
            }

            QueueUniformCudaHipRtImpl(QueueUniformCudaHipRtImpl&&) = default;
            auto operator=(QueueUniformCudaHipRtImpl&&) -> QueueUniformCudaHipRtImpl& = delete;

            AV2P_FN_HOST ~QueueUniformCudaHipRtImpl()
            {
                AV2P_DEBUG_MINIMAL_LOG_SCOPE;

                // Make sure all pending async work is finished before destroying the stream to guarantee determinism.
                // This would not be necessary for plain CUDA/HIP operations, but we can have host functions in the
                // stream, which reference this queue instance and its CallbackThread. Make sure they are done.
                if(m_isOwning)
                {
                    AV2P_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(TApi::streamSynchronize(m_UniformCudaHipQueue));
                    AV2P_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(TApi::streamDestroy(m_UniformCudaHipQueue));
                }
            }

            [[nodiscard]] auto getNativeHandle() const noexcept
            {
                return m_UniformCudaHipQueue;
            }

        public:
            DevUniformCudaHipRt<TApi> const m_dev; //!< The device this queue is bound to.
            core::CallbackThread m_callbackThread;

        private:
            typename TApi::Stream_t m_UniformCudaHipQueue;
            bool m_isOwning;
        };

        //! The CUDA/HIP RT queue.
        template<typename TApi, bool TBlocking>
        class QueueUniformCudaHipRt
            : public interface::Implements<ConceptCurrentThreadWaitFor, QueueUniformCudaHipRt<TApi, TBlocking>>
            , public interface::Implements<ConceptQueue, QueueUniformCudaHipRt<TApi, TBlocking>>
            , public interface::Implements<ConceptGetDev, QueueUniformCudaHipRt<TApi, TBlocking>>
        {
        public:
            AV2P_FN_HOST QueueUniformCudaHipRt(DevUniformCudaHipRt<TApi> const& dev)
                : m_spQueueImpl(std::make_shared<QueueUniformCudaHipRtImpl<TApi>>(dev))
            {
                dev.registerQueue(m_spQueueImpl);
            }

            AV2P_FN_HOST QueueUniformCudaHipRt(typename TApi::Stream_t stream)
                : m_spQueueImpl(std::make_shared<QueueUniformCudaHipRtImpl<TApi>>(stream))
            {
                m_spQueueImpl->m_dev.registerQueue(m_spQueueImpl);
            }

            AV2P_FN_HOST auto operator==(QueueUniformCudaHipRt const& rhs) const -> bool
            {
                return (m_spQueueImpl == rhs.m_spQueueImpl);
            }

            AV2P_FN_HOST auto operator!=(QueueUniformCudaHipRt const& rhs) const -> bool
            {
                return !((*this) == rhs);
            }

            [[nodiscard]] auto getNativeHandle() const noexcept
            {
                return m_spQueueImpl->getNativeHandle();
            }

            auto getCallbackThread() -> core::CallbackThread&
            {
                return m_spQueueImpl->m_callbackThread;
            }

        public:
            std::shared_ptr<QueueUniformCudaHipRtImpl<TApi>> m_spQueueImpl;
        };
    } // namespace uniform_cuda_hip::detail

    namespace trait
    {
        //! The CUDA/HIP RT queue device get trait specialization.
        template<typename TApi, bool TBlocking>
        struct GetDev<uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking>>
        {
            AV2P_FN_HOST static auto getDev(
                uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking> const& queue)
                -> DevUniformCudaHipRt<TApi>
            {
                return queue.m_spQueueImpl->m_dev;
            }
        };

        //! The CUDA/HIP RT queue test trait specialization.
        template<typename TApi, bool TBlocking>
        struct Empty<uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking>>
        {
            AV2P_FN_HOST static auto empty(
                uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking> const& queue) -> bool
            {
                AV2P_DEBUG_MINIMAL_LOG_SCOPE;

                // Query is allowed even for queues on non current device.
                typename TApi::Error_t ret = TApi::success;
                AV2P_UNIFORM_CUDA_HIP_RT_CHECK_IGNORE(
                    ret = TApi::streamQuery(queue.getNativeHandle()),
                    TApi::errorNotReady);
                return (ret == TApi::success);
            }
        };

        //! The CUDA/HIP RT queue thread wait trait specialization.
        //!
        //! Blocks execution of the calling thread until the queue has finished processing all previously requested
        //! tasks (kernels, data copies, ...)
        template<typename TApi, bool TBlocking>
        struct CurrentThreadWaitFor<uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking>>
        {
            AV2P_FN_HOST static auto currentThreadWaitFor(
                uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking> const& queue) -> void
            {
                AV2P_DEBUG_MINIMAL_LOG_SCOPE;

                // Sync is allowed even for queues on non current device.
                AV2P_UNIFORM_CUDA_HIP_RT_CHECK(TApi::streamSynchronize(queue.getNativeHandle()));
            }
        };

        //! The CUDA/HIP RT blocking queue device type trait specialization.
        template<typename TApi, bool TBlocking>
        struct DevType<uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking>>
        {
            using type = DevUniformCudaHipRt<TApi>;
        };

        //! The CUDA/HIP RT blocking queue event type trait specialization.
        template<typename TApi, bool TBlocking>
        struct EventType<uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking>>
        {
            using type = EventUniformCudaHipRt<TApi>;
        };

        //! The CUDA/HIP RT blocking queue enqueue trait specialization.
        template<typename TApi, bool TBlocking, typename TTask>
        struct Enqueue<uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking>, TTask>
        {
            using QueueImpl = uniform_cuda_hip::detail::QueueUniformCudaHipRtImpl<TApi>;

            struct HostFuncData
            {
                // We don't need to keep the queue alive, because in it's dtor it will synchronize with the CUDA/HIP
                // stream and wait until all host functions and the CallbackThread are done. It's actually an error to
                // copy the queue into the host function. Destroying it here would call CUDA/HIP APIs from the host
                // function. Passing it further to the Callback thread, would make the Callback thread hold a task
                // containing the queue with the CallbackThread itself. Destroying the task if no other queue instance
                // exists will make the CallbackThread join itself and crash.
                QueueImpl& q;
                TTask t;
            };

            AV2P_FN_HOST static void uniformCudaHipRtHostFunc(void* arg)
            {
                auto data = std::unique_ptr<HostFuncData>(reinterpret_cast<HostFuncData*>(arg));
                auto& queue = data->q;
                auto f = queue.m_callbackThread.submit([d = std::move(data)] { d->t(); });
                f.wait();
            }

            AV2P_FN_HOST static auto enqueue(
                uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking>& queue,
                TTask const& task) -> void
            {
                AV2P_UNIFORM_CUDA_HIP_RT_CHECK(TApi::launchHostFunc(
                    queue.getNativeHandle(),
                    uniformCudaHipRtHostFunc,
                    new HostFuncData{*queue.m_spQueueImpl, task}));
                if constexpr(TBlocking)
                    AV2P_UNIFORM_CUDA_HIP_RT_CHECK(TApi::streamSynchronize(queue.getNativeHandle()));
            }
        };

        //! The CUDA/HIP RT blocking queue native handle trait specialization.
        template<typename TApi, bool TBlocking>
        struct NativeHandle<uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking>>
        {
            [[nodiscard]] static auto getNativeHandle(
                uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking> const& queue)
            {
                return queue.getNativeHandle();
            }
        };
    } // namespace trait
} // namespace aV2P

#endif

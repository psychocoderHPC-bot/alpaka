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

#include <type_traits>

namespace aV2P
{
    template<typename TDev>
    class EventAlpaka3;

    //! Queue handle wrapping an alpaka 3 `onHost::Queue`.
    //!
    //! `TQueueKind` is the alpaka 3 queue kind (`alpaka::queueKind::Blocking` or
    //! `alpaka::queueKind::NonBlocking`).  The queue kind is honoured at construction time.
    //! `getAlpaka3()` returns a reference into the wrapper and is bound to the wrapper's lifetime.
    template<typename TDev, alpaka::concepts::QueueKind TQueueKind>
    class QueueAlpaka3 final
        : public interface::Implements<ConceptCurrentThreadWaitFor, QueueAlpaka3<TDev, TQueueKind>>
        , public interface::Implements<ConceptQueue, QueueAlpaka3<TDev, TQueueKind>>
        , public interface::Implements<ConceptGetDev, QueueAlpaka3<TDev, TQueueKind>>
    {
    public:
        using QueueType
            = alpaka::onHost::Queue<typename TDev::DevType, alpaka::onHost::QueuePolicyList<TQueueKind>>;

        explicit QueueAlpaka3(TDev const& dev)
            : m_queueImpl{createQueue(dev)}
        {
        }

        //! Non-owning reference to the wrapped alpaka 3 queue.
        [[nodiscard]] auto getAlpaka3() & noexcept -> QueueType&
        {
            return m_queueImpl;
        }

        [[nodiscard]] auto getAlpaka3() const& noexcept -> QueueType const&
        {
            return m_queueImpl;
        }

        auto operator==(QueueAlpaka3 const& rhs) const -> bool
        {
            return m_queueImpl == rhs.m_queueImpl;
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

        QueueType m_queueImpl;
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
#endif

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
                return TDev{queue.getAlpaka3().getDevice()};
            }
        };

        //! Generic fallback: enqueue the legacy task as a host function on the wrapped alpaka 3
        //! queue.  Kernel and memory tasks have more specific specializations.
        template<typename TDev, typename TTask, alpaka::concepts::QueueKind TQueueKind>
        struct Enqueue<QueueAlpaka3<TDev, TQueueKind>, TTask>
        {
            AV2P_FN_HOST static auto enqueue(QueueAlpaka3<TDev, TQueueKind>& queue, TTask const& task) -> void
            {
                queue.getAlpaka3().enqueueHostFn(task);
            }
        };

        template<typename TDev, alpaka::concepts::QueueKind TQueueKind>
        struct Empty<QueueAlpaka3<TDev, TQueueKind>>
        {
            AV2P_FN_HOST static auto empty(QueueAlpaka3<TDev, TQueueKind> const& queue) -> bool
            {
                return queue.getAlpaka3().isEmpty();
            }
        };

        template<typename TDev, alpaka::concepts::QueueKind TQueueKind>
        struct CurrentThreadWaitFor<QueueAlpaka3<TDev, TQueueKind>>
        {
            AV2P_FN_HOST static auto currentThreadWaitFor(QueueAlpaka3<TDev, TQueueKind> const& queue) -> void
            {
                alpaka::onHost::wait(queue.getAlpaka3());
            }
        };
    } // namespace trait
} // namespace aV2P

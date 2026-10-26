/* Copyright 2026 Axel Hübl, Benjamin Worpitz, Matthias Werner, René Widera, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/adapter/Queue.hpp"
#include "aV2P/core/Common.hpp"
#include "aV2P/dev/Traits.hpp"
#include "aV2P/event/Traits.hpp"
#include "aV2P/wait/Traits.hpp"

#include <alpaka/alpaka.hpp>

#include <utility>

namespace aV2P
{
    //! Event handle wrapping an alpaka 3 `onHost::Event`.
    //!
    //! `getAlpaka3()` returns a reference into the wrapper and is bound to the wrapper's lifetime.
    template<typename TDev>
    class EventAlpaka3 final
        : public interface::Implements<ConceptCurrentThreadWaitFor, EventAlpaka3<TDev>>
        , public interface::Implements<ConceptGetDev, EventAlpaka3<TDev>>
    {
    public:
        using EventType = decltype(std::declval<typename TDev::DevType&>().makeEvent());

        //! \param bBusyWaiting Unused; alpaka 3 owns the event synchronisation strategy.
        explicit EventAlpaka3(TDev const& dev, [[maybe_unused]] bool bBusyWaiting = true)
            : m_dev{dev}
            , m_eventImpl{const_cast<typename TDev::DevType&>(m_dev.getAlpaka3()).makeEvent()}
        {
        }

        [[nodiscard]] auto getAlpaka3() & noexcept -> EventType&
        {
            return m_eventImpl;
        }

        [[nodiscard]] auto getAlpaka3() const& noexcept -> EventType const&
        {
            return m_eventImpl;
        }

        auto operator==(EventAlpaka3 const& rhs) const -> bool
        {
            return m_eventImpl == rhs.m_eventImpl;
        }

        auto operator!=(EventAlpaka3 const& rhs) const -> bool
        {
            return !(*this == rhs);
        }

        [[nodiscard]] TDev const& getDev() const noexcept
        {
            return m_dev;
        }

    private:
        TDev m_dev;
        EventType m_eventImpl;
    };

    using EventCpu = EventAlpaka3<DevCpu>;

#ifdef AV2P_ACC_GPU_CUDA_ENABLED
    using EventCudaRt = EventAlpaka3<DevCudaRt>;
#endif

#ifdef AV2P_ACC_GPU_HIP_ENABLED
    using EventHipRt = EventAlpaka3<DevHipRt>;
#endif

#ifdef AV2P_ACC_SYCL_ENABLED
    using EventCpuSycl = EventAlpaka3<DevCpuSycl>;
    using EventGpuSyclIntel = EventAlpaka3<DevGpuSyclIntel>;
#endif

    namespace trait
    {
        template<typename TDev>
        struct DevType<EventAlpaka3<TDev>>
        {
            using type = TDev;
        };

        template<typename TDev>
        struct GetDev<EventAlpaka3<TDev>>
        {
            AV2P_FN_HOST static auto getDev(EventAlpaka3<TDev> const& event) -> TDev
            {
                return event.getDev();
            }
        };

        template<typename TDev>
        struct IsComplete<EventAlpaka3<TDev>>
        {
            AV2P_FN_HOST static auto isComplete(EventAlpaka3<TDev> const& event) -> bool
            {
                return event.getAlpaka3().isComplete();
            }
        };

        template<typename TDev, alpaka::concepts::QueueKind TQueueKind>
        struct EventType<QueueAlpaka3<TDev, TQueueKind>>
        {
            using type = EventAlpaka3<TDev>;
        };

        template<typename TDev, alpaka::concepts::QueueKind TQueueKind>
        struct Enqueue<QueueAlpaka3<TDev, TQueueKind>, EventAlpaka3<TDev>>
        {
            AV2P_FN_HOST static auto enqueue(
                QueueAlpaka3<TDev, TQueueKind>& queue,
                EventAlpaka3<TDev> const& event) -> void
            {
                queue.getAlpaka3().enqueue(event.getAlpaka3());
            }
        };

        template<typename TDev>
        struct CurrentThreadWaitFor<EventAlpaka3<TDev>>
        {
            AV2P_FN_HOST static auto currentThreadWaitFor(EventAlpaka3<TDev> const& event) -> void
            {
                alpaka::onHost::wait(event.getAlpaka3());
            }
        };

        template<typename TDev, alpaka::concepts::QueueKind TQueueKind>
        struct WaiterWaitFor<QueueAlpaka3<TDev, TQueueKind>, EventAlpaka3<TDev>>
        {
            AV2P_FN_HOST static auto waiterWaitFor(
                QueueAlpaka3<TDev, TQueueKind>& queue,
                EventAlpaka3<TDev> const& event) -> void
            {
                queue.getAlpaka3().waitFor(event.getAlpaka3());
            }
        };
    } // namespace trait
} // namespace aV2P

/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

//! Test-only queue used by the legacy queue/event tests.
//!
//! The original 2.2 implementation was an *implementation detail* test double: it derived from the
//! removed `cpu::ICpuQueue` and ran kernel/copy tasks through the old OpenMP backend task objects.
//! aV2P has no public equivalent for executing a kernel collectively inside a user-created OpenMP
//! parallel region, so this compatibility shim owns a regular `QueueCpuBlocking` and delegates every
//! task to it.  Outside an OpenMP parallel region the behaviour is identical to the original class
//! (documented in 2.2).  Inside a parallel region exactly one thread of the team submits the task to
//! the blocking queue, while the other threads synchronise on the same `wait`/`empty` contract.  The
//! legacy collective distribution across the user's OpenMP team is therefore not reproduced; the
//! generic queue/event assertions still hold and the collective-specific tests document this
//! limitation.
#include "aV2P/alpaka.hpp"
#include "aV2P/test/event/EventHostManualTrigger.hpp"
#include "aV2P/test/queue/Queue.hpp"

#include <atomic>
#include <mutex>

#ifdef AV2P_ACC_CPU_B_OMP2_T_SEQ_ENABLED

#    if _OPENMP < 200203
#        error If AV2P_ACC_CPU_B_OMP2_T_SEQ_ENABLED is set, the compiler has to support OpenMP 2.0 or higher!
#    endif

#    include <omp.h>

namespace aV2P
{
    namespace cpu::detail
    {
        //! State shared by all copies of the test queue.
        struct QueueCpuOmp2CollectiveImpl
        {
            explicit QueueCpuOmp2CollectiveImpl(DevCpu const& dev) noexcept
                : m_dev(dev)
                , blockingQueue(dev)
            {
            }

            void busyWaitUntilBlockingQueueEmpty()
            {
                while(!empty(blockingQueue))
                    ;
            }

            DevCpu const m_dev; //!< The device this queue is bound to.
            std::mutex mutable m_mutex;
            QueueCpuBlocking blockingQueue;
            std::atomic<uint32_t> m_uCurrentlyExecutingTask = 0;
        };
    } // namespace cpu::detail

    //! The CPU collective device queue compatibility shim (see the file comment).
    struct QueueCpuOmp2Collective final : interface::Implements<ConceptCurrentThreadWaitFor, QueueCpuOmp2Collective>
    {
        explicit QueueCpuOmp2Collective(DevCpu const& dev)
            : m_spQueueImpl(std::make_shared<cpu::detail::QueueCpuOmp2CollectiveImpl>(dev))
        {
        }

        auto operator==(QueueCpuOmp2Collective const& rhs) const -> bool
        {
            return m_spQueueImpl == rhs.m_spQueueImpl;
        }

        auto operator!=(QueueCpuOmp2Collective const& rhs) const -> bool
        {
            return !((*this) == rhs);
        }

        std::shared_ptr<cpu::detail::QueueCpuOmp2CollectiveImpl> m_spQueueImpl;
    };

    namespace trait
    {
        //! The CPU blocking device queue device type trait specialization.
        template<>
        struct DevType<QueueCpuOmp2Collective>
        {
            using type = DevCpu;
        };

        //! The CPU blocking device queue device get trait specialization.
        template<>
        struct GetDev<QueueCpuOmp2Collective>
        {
            AV2P_FN_HOST static auto getDev(QueueCpuOmp2Collective const& queue) -> DevCpu
            {
                return queue.m_spQueueImpl->m_dev;
            }
        };

        //! The CPU blocking device queue event type trait specialization.
        template<>
        struct EventType<QueueCpuOmp2Collective>
        {
            using type = EventCpu;
        };

        //! Submit the task to the wrapped blocking queue.
        //!
        //! Outside an OpenMP parallel region this is exactly the 2.2 behaviour. Inside a parallel
        //! region one thread of the team submits the task; the remaining threads keep synchronising
        //! through `wait`/`empty`. The task object is forwarded unchanged, so kernel, copy and event
        //! tasks all use their adapter enqueue specialisations.
        template<typename TTask>
        struct Enqueue<QueueCpuOmp2Collective, TTask>
        {
            AV2P_FN_HOST static auto enqueue(QueueCpuOmp2Collective& queue, TTask const& task) -> void
            {
                if(::omp_in_parallel() != 0)
                {
                    // wait for all tasks enqueued before the parallel region
                    queue.m_spQueueImpl->busyWaitUntilBlockingQueueEmpty();
                    ++queue.m_spQueueImpl->m_uCurrentlyExecutingTask;
#    pragma omp single nowait
                    {
                        std::lock_guard<std::mutex> lk(queue.m_spQueueImpl->m_mutex);
                        aV2P::enqueue(queue.m_spQueueImpl->blockingQueue, task);
                    }
                    --queue.m_spQueueImpl->m_uCurrentlyExecutingTask;
                }
                else
                {
                    std::lock_guard<std::mutex> lk(queue.m_spQueueImpl->m_mutex);
                    aV2P::enqueue(queue.m_spQueueImpl->blockingQueue, task);
                }
            }
        };

        //! EventHostManualTriggerCpu is not supported together with the collective queue, but the
        //! event tests instantiate the enqueue for every queue type. Mirrors the 2.2 no-op.
        template<>
        struct Enqueue<QueueCpuOmp2Collective, test::EventHostManualTriggerCpu<>>
        {
            AV2P_FN_HOST static auto enqueue(QueueCpuOmp2Collective&, test::EventHostManualTriggerCpu<>&) -> void
            {
            }
        };

        //! The CPU blocking device queue test trait specialization.
        template<>
        struct Empty<QueueCpuOmp2Collective>
        {
            AV2P_FN_HOST static auto empty(QueueCpuOmp2Collective const& queue) -> bool
            {
                return queue.m_spQueueImpl->m_uCurrentlyExecutingTask == 0u
                       && aV2P::empty(queue.m_spQueueImpl->blockingQueue);
            }
        };

        //! The CPU blocking device queue thread wait trait specialization.
        template<>
        struct CurrentThreadWaitFor<QueueCpuOmp2Collective>
        {
            AV2P_FN_HOST static auto currentThreadWaitFor(QueueCpuOmp2Collective const& queue) -> void
            {
                if(::omp_in_parallel() != 0)
                {
                    // wait for all tasks enqueued before the parallel region
                    queue.m_spQueueImpl->busyWaitUntilBlockingQueueEmpty();
#    pragma omp barrier
                }
                else
                {
                    std::lock_guard<std::mutex> lk(queue.m_spQueueImpl->m_mutex);
                    wait(queue.m_spQueueImpl->blockingQueue);
                }
            }
        };

        //! The CPU OpenMP2 collective queue event wait trait specialization.
        template<>
        struct WaiterWaitFor<QueueCpuOmp2Collective, EventCpu>
        {
            AV2P_FN_HOST static auto waiterWaitFor(QueueCpuOmp2Collective& queue, EventCpu const& event) -> void
            {
                if(::omp_in_parallel() != 0)
                {
                    queue.m_spQueueImpl->busyWaitUntilBlockingQueueEmpty();
                    wait(queue);
                }
                else
                    wait(queue.m_spQueueImpl->blockingQueue, event);
            }
        };
    } // namespace trait

    //! The blocking queue trait specialization for a OpenMP2 collective CPU queue.
    template<>
    struct test::trait::IsBlockingQueue<QueueCpuOmp2Collective> : std::true_type
    {
    };
} // namespace aV2P

#endif

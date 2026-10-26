/* Copyright 2022 Axel Huebl, Benjamin Worpitz, René Widera, Bernhard Manfred Gruber, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/event/Traits.hpp>
#include <aV2P/test/event/EventHostManualTrigger.hpp>
#include <aV2P/test/queue/Queue.hpp>
#include <aV2P/test/queue/QueueCpuOmp2Collective.hpp>
#include <aV2P/test/queue/QueueTestFixture.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

using TestQueues = aV2P::meta::Concatenate<
    aV2P::test::TestQueues
#ifdef AV2P_ACC_CPU_B_OMP2_T_SEQ_ENABLED
    ,
    std::tuple<std::tuple<aV2P::DevCpu, aV2P::QueueCpuOmp2Collective>>
#endif
    >;

TEMPLATE_LIST_TEST_CASE("eventTestShouldInitiallyBeTrue", "[event]", TestQueues)
{
    using DevQueue = TestType;
    using Fixture = aV2P::test::QueueTestFixture<DevQueue>;
    using Queue = typename Fixture::Queue;

    Fixture f;
    aV2P::Event<Queue> event(f.m_dev);

    REQUIRE(aV2P::isComplete(event));
}

TEMPLATE_LIST_TEST_CASE("eventTestShouldBeFalseWhileInQueueAndTrueAfterBeingProcessed", "[event]", TestQueues)
{
    using DevQueue = TestType;
    using Fixture = aV2P::test::QueueTestFixture<DevQueue>;
    using Queue = typename Fixture::Queue;
    using Dev = typename Fixture::Dev;

    Fixture f1;
    if(aV2P::test::isEventHostManualTriggerSupported(f1.m_dev))
    {
        auto q1 = f1.m_queue;
        aV2P::Event<Queue> e1(f1.m_dev);
        aV2P::test::EventHostManualTrigger<Dev> k1(f1.m_dev);

        if(!aV2P::test::IsBlockingQueue<Queue>::value)
        {
            aV2P::enqueue(q1, k1);
        }

        aV2P::enqueue(q1, e1);

        if(!aV2P::test::IsBlockingQueue<Queue>::value)
        {
            REQUIRE(aV2P::isComplete(e1) == false);

            k1.trigger();

            aV2P::wait(q1);
        }

        REQUIRE(aV2P::isComplete(e1));
    }
    else
    {
        std::cerr << "Cannot execute test because CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS is not supported!"
                  << std::endl;
    }
}

TEMPLATE_LIST_TEST_CASE("eventReEnqueueShouldBePossibleIfNobodyWaitsFor", "[event]", TestQueues)
{
    using DevQueue = TestType;
    using Fixture = aV2P::test::QueueTestFixture<DevQueue>;
    using Queue = typename Fixture::Queue;
    using Dev = typename Fixture::Dev;

    if(!aV2P::test::IsBlockingQueue<Queue>::value)
    {
        Fixture f1;
        if(aV2P::test::isEventHostManualTriggerSupported(f1.m_dev))
        {
            auto q1 = f1.m_queue;
            aV2P::Event<Queue> e1(f1.m_dev);
            aV2P::test::EventHostManualTrigger<Dev> k1(f1.m_dev);
            aV2P::test::EventHostManualTrigger<Dev> k2(f1.m_dev);

            // q1 = [k1]
            aV2P::enqueue(q1, k1);
            REQUIRE(!aV2P::isComplete(k1));

            // q1 = [k1, e1]
            aV2P::enqueue(q1, e1);
            REQUIRE(!aV2P::isComplete(k1));
            REQUIRE(!aV2P::isComplete(e1));

            // q1 = [k1, e1, k2]
            aV2P::enqueue(q1, k2);
            REQUIRE(!aV2P::isComplete(k1));
            REQUIRE(!aV2P::isComplete(e1));
            REQUIRE(!aV2P::isComplete(k2));

            // re-enqueue should be possible
            // q1 = [k1, k2, e1]
            aV2P::enqueue(q1, e1);
            REQUIRE(!aV2P::isComplete(k1));
            REQUIRE(!aV2P::isComplete(k2));
            REQUIRE(!aV2P::isComplete(e1));

            // q1 = [k2, e1]
            k1.trigger();
            REQUIRE(aV2P::isComplete(k1));
            REQUIRE(!aV2P::isComplete(k2));
            REQUIRE(!aV2P::isComplete(e1));

            // q1 = [e1]
            k2.trigger();
            REQUIRE(aV2P::isComplete(k2));
            aV2P::wait(e1);
            REQUIRE(aV2P::isComplete(e1));
        }
        else
        {
            std::cerr << "Cannot execute test because CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS is not supported by "
                         "the device!"
                      << std::endl;
        }
    }
}

TEMPLATE_LIST_TEST_CASE("eventReEnqueueShouldBePossibleIfSomeoneWaitsFor", "[event]", TestQueues)
{
    using DevQueue = TestType;
    using Fixture = aV2P::test::QueueTestFixture<DevQueue>;
    using Queue = typename Fixture::Queue;
    using Dev = typename Fixture::Dev;

    if(!aV2P::test::IsBlockingQueue<Queue>::value)
    {
        Fixture f1;
        Fixture f2;
        if(aV2P::test::isEventHostManualTriggerSupported(f1.m_dev)
           && aV2P::test::isEventHostManualTriggerSupported(f2.m_dev))
        {
            auto q1 = f1.m_queue;
            auto q2 = f2.m_queue;
            aV2P::Event<Queue> e1(f1.m_dev);
            aV2P::Event<Queue> e2(f2.m_dev);
            aV2P::test::EventHostManualTrigger<Dev> k1(f1.m_dev);
            aV2P::test::EventHostManualTrigger<Dev> k2(f1.m_dev);

            aV2P::enqueue(q1, k1);

            // q1 = [k1]
            REQUIRE(!aV2P::isComplete(k1));

            aV2P::enqueue(q1, e1);

            // q1 = [k1, e1]
            REQUIRE(!aV2P::isComplete(k1));
            REQUIRE(!aV2P::isComplete(e1));

            aV2P::enqueue(q1, k2);

            // q1 = [k1, e1, k2]
            REQUIRE(!aV2P::isComplete(k1));
            REQUIRE(!aV2P::isComplete(e1));
            REQUIRE(!aV2P::isComplete(k2));

            // wait for e1
            aV2P::wait(q2, e1);
            // q2 = [->e1]

            aV2P::enqueue(q2, e2);

            // q2 = [->e1, e2]
            REQUIRE(!aV2P::isComplete(e2));

            // re-enqueue should be possible
            aV2P::enqueue(q1, e1);

            // q1 = [k1, e1, k2, e1_new]
            // q2 = [->e1, e2]
            REQUIRE(!aV2P::isComplete(k1));
            REQUIRE(!aV2P::isComplete(k2));
            REQUIRE(!aV2P::isComplete(e1));
            REQUIRE(!aV2P::isComplete(e2));

            k1.trigger();

            // q1 = [k2, e1_new]
            // q2 = []
            REQUIRE(aV2P::isComplete(k1));
            REQUIRE(!aV2P::isComplete(k2));
            REQUIRE(!aV2P::isComplete(e1));
            REQUIRE(aV2P::isComplete(e2));

            k2.trigger();

            // q1 = []
            // q2 = []
            REQUIRE(aV2P::isComplete(k2));
            aV2P::wait(e1);
            REQUIRE(aV2P::isComplete(e1));
            aV2P::wait(e2);
            REQUIRE(aV2P::isComplete(e2));
        }
        else
        {
            std::cerr << "Cannot execute test because CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS is not supported by "
                         "the device!"
                      << std::endl;
        }
    }
}

// github issue #388
TEMPLATE_LIST_TEST_CASE("waitForEventThatAlreadyFinishedShouldBeSkipped", "[event]", TestQueues)
{
    using DevQueue = TestType;
    using Fixture = aV2P::test::QueueTestFixture<DevQueue>;
    using Queue = typename Fixture::Queue;
    using Dev = typename Fixture::Dev;

    if(!aV2P::test::IsBlockingQueue<Queue>::value)
    {
        Fixture f1;
        Fixture f2;
        if(aV2P::test::isEventHostManualTriggerSupported(f1.m_dev)
           && aV2P::test::isEventHostManualTriggerSupported(f2.m_dev))
        {
            auto q1 = f1.m_queue;
            auto q2 = f2.m_queue;
            aV2P::test::EventHostManualTrigger<Dev> k1(f1.m_dev);
            aV2P::test::EventHostManualTrigger<Dev> k2(f2.m_dev);
            aV2P::Event<Queue> e1(f1.m_dev);

            // 1. kernel k1 is enqueued into queue q1
            aV2P::enqueue(q1, k1);
            // q1 = [k1]

            // 2. kernel k2 is enqueued into queue q2
            aV2P::enqueue(q2, k2);
            // q2 = [k2]

            // 3. event e1 is enqueued into queue q1
            aV2P::enqueue(q1, e1);
            // q1 = [k1, e1]

            // 4. q2 waits for e1
            aV2P::wait(q2, e1);
            // q2 = [k2, ->e1]

            // 5. kernel k1 finishes
            k1.trigger();
            // q1 = [e1]

            // 6. e1 is finished
            aV2P::wait(e1);

            // q1 = []
            // q2 = [k2, ->e1]
            REQUIRE(!aV2P::isComplete(k2));
            REQUIRE(aV2P::isComplete(e1));

            // 7. e1 is re-enqueued again but this time into q2
            aV2P::enqueue(q2, e1);

            // q2 = [k2, ->e1, e1]
            REQUIRE(!aV2P::isComplete(k2));
            REQUIRE(!aV2P::isComplete(e1));

            // 8. kernel k2 finishes
            k2.trigger();
            // q2 = []
            REQUIRE(aV2P::isComplete(e1));

            // 9. e1 had already been signaled so there should not be waited even though the event is now reused within
            // q2 and its current state is 'unfinished' again. q2 = [e1]

            // Both queues should successfully finish
            aV2P::wait(q1);
            aV2P::wait(q2);
        }
        else
        {
            std::cerr << "Cannot execute test because CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS is not supported by "
                         "the device!"
                      << std::endl;
        }
    }
}

TEMPLATE_LIST_TEST_CASE("eventReEnqueueWithSomeoneWaitsForEventInOrderLifetimeRelease", "[event]", TestQueues)
{
    // A re-enqueued event will be released in the order it is recorded. The tests validate that dependencies between
    // queues and the re-enqueued event will be correct.
    using DevQueue = TestType;
    using Fixture = aV2P::test::QueueTestFixture<DevQueue>;
    using Queue = typename Fixture::Queue;
    using Dev = typename Fixture::Dev;

    if(!aV2P::test::IsBlockingQueue<Queue>::value)
    {
        Fixture f1;
        Fixture f2;
        Fixture f3;
        if(aV2P::test::isEventHostManualTriggerSupported(f1.m_dev)
           && aV2P::test::isEventHostManualTriggerSupported(f2.m_dev)
           && aV2P::test::isEventHostManualTriggerSupported(f3.m_dev))
        {
            auto q1 = f1.m_queue;
            auto q2 = f2.m_queue;
            auto q3 = f3.m_queue;

            aV2P::Event<Queue> e1(f1.m_dev);
            aV2P::Event<Queue> e2(f2.m_dev);
            aV2P::Event<Queue> e3(f3.m_dev);
            aV2P::test::EventHostManualTrigger<Dev> k1_0(f1.m_dev);
            aV2P::test::EventHostManualTrigger<Dev> k1_1(f1.m_dev);
            aV2P::test::EventHostManualTrigger<Dev> k2(f2.m_dev);
            aV2P::test::EventHostManualTrigger<Dev> k3(f3.m_dev);

            aV2P::enqueue(q1, k1_0);
            aV2P::enqueue(q1, e1);
            // q1 = [k1_0,e1]
            aV2P::enqueue(q2, k2);
            // q2 = [k2]
            REQUIRE(!aV2P::isComplete(k1_0));

            aV2P::wait(q2, e1);
            aV2P::enqueue(q2, e2);
            // q2 = [k2,->e1,e2]
            aV2P::enqueue(q1, k1_1);
            aV2P::enqueue(q1, e1);
            // q1 = [k1_0,e1,k1_1,e1_new]
            aV2P::enqueue(q3, k3);
            aV2P::wait(q3, e1);
            aV2P::enqueue(q3, e3);

            // q1 = [k1_0,e1,k1_1,e1_new]
            // q2 = [k2,->e1,e2]
            // q3 = [k3,->e1_new,e3]
            REQUIRE(!aV2P::isComplete(k1_0));
            REQUIRE(!aV2P::isComplete(k1_1));
            REQUIRE(!aV2P::isComplete(k2));
            REQUIRE(!aV2P::isComplete(k3));
            REQUIRE(!aV2P::isComplete(e1));
            REQUIRE(!aV2P::isComplete(e2));
            REQUIRE(!aV2P::isComplete(e3));

            k3.trigger();

            // q1 = [k1_0,e1,k1_1,e1_new]
            // q2 = [k2,->e1,e2]
            // q3 = [->e1_new,e3]
            REQUIRE(!aV2P::isComplete(k1_0));
            REQUIRE(!aV2P::isComplete(k1_1));
            REQUIRE(!aV2P::isComplete(k2));
            REQUIRE(aV2P::isComplete(k3));
            REQUIRE(!aV2P::isComplete(e1));
            REQUIRE(!aV2P::isComplete(e2));
            REQUIRE(!aV2P::isComplete(e3));

            k2.trigger();

            // q1 = [k1_0,e1,k1_1,e1_new]
            // q2 = [->e1,e2]
            // q3 = [->e1_new,e3]
            REQUIRE(!aV2P::isComplete(k1_0));
            REQUIRE(!aV2P::isComplete(k1_1));
            REQUIRE(aV2P::isComplete(k2));
            REQUIRE(aV2P::isComplete(k3));
            REQUIRE(!aV2P::isComplete(e1));
            REQUIRE(!aV2P::isComplete(e2));
            REQUIRE(!aV2P::isComplete(e3));

            // After the kernel k1_0 is released e3 is not allowed to be ready because q3 depends on the oldest e1_new
            // state.
            k1_0.trigger();
            // q1 = [k1_1,e1_new]
            // q2 = []
            // q3 = [->e1_new,e3]

            REQUIRE(aV2P::isComplete(k1_0));
            REQUIRE(!aV2P::isComplete(k1_1));
            REQUIRE(aV2P::isComplete(k2));
            REQUIRE(aV2P::isComplete(k3));
            REQUIRE(!aV2P::isComplete(e1));
            REQUIRE(aV2P::isComplete(e2));
            REQUIRE(!aV2P::isComplete(e3));

            k1_1.trigger();
            // q1 = []
            // q2 = []
            // q3 = []
            REQUIRE(aV2P::isComplete(k1_0));
            REQUIRE(aV2P::isComplete(k1_1));
            REQUIRE(aV2P::isComplete(k2));
            REQUIRE(aV2P::isComplete(k3));
            REQUIRE(aV2P::isComplete(e1));
            REQUIRE(aV2P::isComplete(e2));
            REQUIRE(aV2P::isComplete(e3));
        }
        else
        {
            std::cerr << "Cannot execute test because CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS is not supported by "
                         "the device!"
                      << std::endl;
        }
    }
}

TEMPLATE_LIST_TEST_CASE("eventReEnqueueWithSomeoneWaitsForEventOutOfOrderLifetimeRelease", "[event]", TestQueues)
{
    // A re-enqueued event will be released in the opposite order it is recorded.
    // The tests validate that dependencies between queues and the re-enqueued event will be correct.
    using DevQueue = TestType;
    using Fixture = aV2P::test::QueueTestFixture<DevQueue>;
    using Queue = typename Fixture::Queue;
    using Dev = typename Fixture::Dev;

    if(!aV2P::test::IsBlockingQueue<Queue>::value)
    {
        Fixture f1;
        Fixture f2;
        Fixture f3;
        if(aV2P::test::isEventHostManualTriggerSupported(f1.m_dev)
           && aV2P::test::isEventHostManualTriggerSupported(f2.m_dev))
        {
            auto q1 = f1.m_queue;
            auto q2 = f2.m_queue;
            auto q3 = f3.m_queue;
            aV2P::Event<Queue> e1(f1.m_dev);
            aV2P::Event<Queue> e2(f2.m_dev);
            aV2P::Event<Queue> e3(f3.m_dev);
            aV2P::test::EventHostManualTrigger<Dev> k1_0(f1.m_dev);
            aV2P::test::EventHostManualTrigger<Dev> k2(f2.m_dev);

            aV2P::enqueue(q1, k1_0);
            aV2P::enqueue(q1, e1);
            // q1 = [k1_0,e1]
            aV2P::enqueue(q2, k2);
            // q2 = [k2]
            REQUIRE(!aV2P::isComplete(k1_0));

            aV2P::wait(q3, e1);
            aV2P::enqueue(q3, e3);
            // q3 = [->e1,e3]

            aV2P::enqueue(q2, e1);
            // q2 = [k2,e1_new]

            aV2P::enqueue(q2, e2);
            // q1 = [k1_0,e1]
            // q2 = [k2,e1_new,e2]
            // q3 = [->e1,e3]

            REQUIRE(!aV2P::isComplete(k1_0));
            REQUIRE(!aV2P::isComplete(k2));
            REQUIRE(!aV2P::isComplete(e1));
            REQUIRE(!aV2P::isComplete(e2));
            REQUIRE(!aV2P::isComplete(e3));

            // We release first the kernel which is blocking the most recent enqueue of event e1.
            // q3 is not allowed to be freed because this queue depends on the oldest enqueue of e1.
            k2.trigger();

            // q1 = [k1_0,e1]
            // q2 = []
            // q3 = [->e1,e3]

            REQUIRE(!aV2P::isComplete(k1_0));

            REQUIRE(aV2P::isComplete(k2));
            REQUIRE(aV2P::isComplete(e1));
            REQUIRE(aV2P::isComplete(e2));
            REQUIRE(!aV2P::isComplete(e3));


            k1_0.trigger();

            // q1 = []
            // q2 = []
            // q3 = []
            REQUIRE(aV2P::isComplete(k1_0));

            REQUIRE(aV2P::isComplete(k2));
            REQUIRE(aV2P::isComplete(e1));
            REQUIRE(aV2P::isComplete(e2));
            REQUIRE(aV2P::isComplete(e3));

            aV2P::wait(q1);
            aV2P::wait(q2);
            aV2P::wait(q3);
        }
        else
        {
            std::cerr << "Cannot execute test because CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS is not supported by "
                         "the device!"
                      << std::endl;
        }
    }
}

/* Copyright 2026 Axel Hübl, Benjamin Worpitz, Bernhard Manfred Gruber, Jan Stephan, Simone Balducci
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/adapter/Api.hpp>
#include <aV2P/meta/Concatenate.hpp>
#include <aV2P/queue/Traits.hpp>
#include <aV2P/test/queue/Queue.hpp>
#include <aV2P/test/queue/QueueCpuOmp2Collective.hpp>
#include <aV2P/test/queue/QueueTestFixture.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <future>
#include <thread>
#include <typeinfo>

using TestQueues = aV2P::meta::Concatenate<
    aV2P::test::TestQueues
#ifdef AV2P_ACC_CPU_B_OMP2_T_SEQ_ENABLED
    ,
    std::tuple<std::tuple<aV2P::DevCpu, aV2P::QueueCpuOmp2Collective>>
#endif
    >;

TEMPLATE_LIST_TEST_CASE("queueIsInitiallyEmpty", "[queue]", TestQueues)
{
    using DevQueue = TestType;
    using Fixture = aV2P::test::QueueTestFixture<DevQueue>;
    Fixture f;

    CHECK(aV2P::empty(f.m_queue));
}

TEMPLATE_LIST_TEST_CASE("queueCallbackIsWorking", "[queue]", TestQueues)
{
    using DevQueue = TestType;
    using Fixture = aV2P::test::QueueTestFixture<DevQueue>;
    Fixture f;

    // Lambda callable
    {
        std::promise<bool> promise;
        aV2P::enqueue(f.m_queue, [&]() { promise.set_value(true); });
        CHECK(promise.get_future().get());
    }

    // std::function rvalue
    {
        std::promise<bool> promise;
        std::function<void()> callback = [&]() { promise.set_value(true); };
        aV2P::enqueue(f.m_queue, std::move(callback));
        CHECK(promise.get_future().get());
    }

    // std::function lvalue
    {
        std::promise<bool> promise;
        std::function<void()> callback = [&]() { promise.set_value(true); };
        aV2P::enqueue(f.m_queue, callback);
        CHECK(promise.get_future().get());
    }
}

TEMPLATE_LIST_TEST_CASE("queueTaskValueCategories", "[queue]", TestQueues)
{
    using DevQueue = TestType;
    using Fixture = aV2P::test::QueueTestFixture<DevQueue>;
    Fixture f;

    bool flag = false;
    auto task = [&] { flag = true; };
    aV2P::enqueue(f.m_queue, task); // lvalue
    aV2P::wait(f.m_queue);
    CHECK(flag == true);

    flag = false;
    aV2P::enqueue(f.m_queue, std::move(task)); // xvalue
    aV2P::wait(f.m_queue);
    CHECK(flag == true);

    flag = false;
    aV2P::enqueue(f.m_queue, [&] { flag = true; }); // prvalue
    aV2P::wait(f.m_queue);
    CHECK(flag == true);
}

TEMPLATE_LIST_TEST_CASE("queueWaitShouldWork", "[queue]", TestQueues)
{
    using DevQueue = TestType;
    using Fixture = aV2P::test::QueueTestFixture<DevQueue>;
    Fixture f;

    std::atomic<bool> callbackFinished{false};
    aV2P::enqueue(
        f.m_queue,
        [&callbackFinished]() noexcept
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100u));
            callbackFinished = true;
        });

    aV2P::wait(f.m_queue);
    CHECK(callbackFinished.load() == true);
}

TEMPLATE_LIST_TEST_CASE(
    "queueShouldNotBeEmptyWhenLastTaskIsStillExecutingAndIsEmptyAfterProcessingFinished",
    "[queue]",
    TestQueues)
{
    using DevQueue = TestType;
    using Fixture = aV2P::test::QueueTestFixture<DevQueue>;
    Fixture f;

    std::atomic<bool> callbackFinished{false};
    std::atomic<bool> inKernelIsNotEmptyBeforeSleep{false};
    std::atomic<bool> inKernelIsNotEmptyAfterSleep{false};

    aV2P::enqueue(
        f.m_queue,
        [&]() noexcept
        {
            // This callback function is in the queue therefore the queue can not be empty.
            inKernelIsNotEmptyBeforeSleep = !aV2P::empty(f.m_queue);
            std::this_thread::sleep_for(std::chrono::milliseconds(100u));
            // Check again, the queue must stay not empty.
            inKernelIsNotEmptyAfterSleep = !aV2P::empty(f.m_queue);
            callbackFinished = true;
        });

    // A non-blocking queue will always stay empty because the task has been executed immediately.
    if(!aV2P::test::IsBlockingQueue<typename Fixture::Queue>::value)
    {
        aV2P::wait(f.m_queue);
    }

    CHECK(callbackFinished.load() == true);
    CHECK(inKernelIsNotEmptyBeforeSleep.load() == true);
    CHECK(inKernelIsNotEmptyAfterSleep.load() == true);

    // The blocking queue must be empty because we wait for the non-blocking queue and the blocking on is synchronized
    // implicitly.
    CHECK(aV2P::empty(f.m_queue));
}

TEMPLATE_LIST_TEST_CASE("taskIsDestroyedAfterExecution", "[queue]", TestQueues)
{
    struct Kernel
    {
        std::atomic<bool>* destroyed;

        explicit Kernel(std::atomic<bool>& a) : destroyed(&a)
        {
        }

        Kernel(Kernel const&) = default;

        ~Kernel()
        {
            *destroyed = true;
        }

        void operator()() const
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1u));
        }
    };

    using DevQueue = TestType;
    auto f = aV2P::test::QueueTestFixture<DevQueue>{};

    std::atomic<bool> destroyed{false};
    aV2P::enqueue(f.m_queue, Kernel{destroyed});

    if(!aV2P::test::IsBlockingQueue<decltype(f.m_queue)>::value)
        aV2P::wait(f.m_queue);
    CHECK(destroyed == true);
    CHECK(aV2P::empty(f.m_queue));
}

TEMPLATE_LIST_TEST_CASE("queueShouldNotExecuteTasksInParallel", "[queue]", TestQueues)
{
    using DevQueue = TestType;
    using Fixture = aV2P::test::QueueTestFixture<DevQueue>;
    Fixture f;

    std::atomic<bool> taskIsExecuting(false);
    std::promise<void> firstTaskFinished;
    std::future<void> firstTaskFinishedFuture = firstTaskFinished.get_future();
    std::atomic<bool> firstTaskInKernelIsExecutingBeforeSleep{false};
    std::atomic<bool> firstTaskInKernelIsExecutingAfterSleep{false};
    std::promise<void> secondTaskFinished;
    std::future<void> secondTaskFinishedFuture = secondTaskFinished.get_future();
    std::atomic<bool> secondTaskInKernelIsExecutingBeforeSleep{false};
    std::atomic<bool> secondTaskInKernelIsExecutingAfterSleep{false};

    std::thread thread1(
        [&]()
        {
            aV2P::enqueue(
                f.m_queue,
                [&]() noexcept
                {
                    firstTaskInKernelIsExecutingBeforeSleep = !taskIsExecuting.exchange(true);
                    std::this_thread::sleep_for(std::chrono::milliseconds(100u));
                    firstTaskInKernelIsExecutingAfterSleep = taskIsExecuting.exchange(false);
                    firstTaskFinished.set_value();
                });
        });

    std::thread thread2(
        [&]()
        {
            aV2P::enqueue(
                f.m_queue,
                [&]() noexcept
                {
                    secondTaskInKernelIsExecutingBeforeSleep = !taskIsExecuting.exchange(true);
                    std::this_thread::sleep_for(std::chrono::milliseconds(100u));
                    secondTaskInKernelIsExecutingAfterSleep = taskIsExecuting.exchange(false);
                    secondTaskFinished.set_value();
                });
        });

    // Both tasks have to be enqueued
    thread1.join();
    thread2.join();

    aV2P::wait(f.m_queue);
    CHECK(firstTaskInKernelIsExecutingBeforeSleep.load() == true);
    CHECK(firstTaskInKernelIsExecutingAfterSleep.load() == true);
    CHECK(secondTaskInKernelIsExecutingBeforeSleep.load() == true);
    CHECK(secondTaskInKernelIsExecutingAfterSleep.load() == true);

    firstTaskFinishedFuture.get();
    secondTaskFinishedFuture.get();
}

//! This task launches a long task in a non-blocking queue and destroys the
//! queue before the task is finished. The run time of the task is a bit longer
//! than all other tests here, so the task would run past program termination if
//! not synchronized somewhere. This test has no assertion, because the tested
//! behaviour is outside of any function scope.
//!
//! A thread running past program termination invokes undefined behavior, which
//! occasionally leads to crashes after termination. I.e. this test may crash
//! after catch2 reported success.
TEMPLATE_LIST_TEST_CASE("nonBlockingQueueShouldNotRunPastProgramTermination", "[queue]", TestQueues)
{
    using DevQueue = TestType;
    using Fixture = aV2P::test::QueueTestFixture<DevQueue>;
    Fixture f;

    if(!aV2P::test::IsBlockingQueue<typename Fixture::Queue>::value)
    {
        // enqueue long task, destroy queue, see what happens
        aV2P::enqueue(
            f.m_queue,
            [name = aV2P::core::demangled<decltype(f.m_queue)>]
            {
                std::cout << "BEGIN not-awaited task in Queue '" << name
                          << "' (if there is no matching 'END' line, the task ran past program termination)"
                          << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(3000u));
                std::cout << "END not-awaited task in Queue '" << name << "'" << std::endl;
            });
    }
}

TEMPLATE_LIST_TEST_CASE("enqueueBenchmark", "[queue]", aV2P::test::TestQueues)
{
    using DevQueue = TestType;
    using Fixture = aV2P::test::QueueTestFixture<DevQueue>;
    Fixture f;

    constexpr auto reps = 1000;
    BENCHMARK("Enqueue " + std::to_string(reps))
    {
        std::atomic<int> count = 0;
        for(int i = 0; i < reps; i++)
            aV2P::enqueue(f.m_queue, [&]() noexcept { ++count; });
        aV2P::wait(f.m_queue);
        return count.load();
    };
}

TEMPLATE_LIST_TEST_CASE("concepts::Queue", "[queue]", aV2P::test::TestQueues)
{
    using DevQueue = TestType;
    using Fixture = aV2P::test::QueueTestFixture<DevQueue>;
    Fixture f;

    REQUIRE(aV2P::concepts::Queue<decltype(f.m_queue)>);
}

#if defined(AV2P_ACC_GPU_CUDA_ENABLED) || defined(AV2P_ACC_GPU_HIP_ENABLED)

#    if defined(AV2P_ACC_GPU_CUDA_ENABLED)
using TApi = aV2P::ApiCudaRt;
#    elif defined(AV2P_ACC_GPU_HIP_ENABLED)
using TApi = aV2P::ApiHipRt;
#    endif

TEMPLATE_LIST_TEST_CASE("constructQueueWithStream", "[queue]", aV2P::test::TestQueues)
{
    TApi::Stream_t stream;
    AV2P_UNIFORM_CUDA_HIP_RT_CHECK(TApi::streamCreate(&stream));
    auto queue = aV2P::QueueUniformCudaHipRtBlocking<TApi>(stream);
    AV2P_UNIFORM_CUDA_HIP_RT_CHECK(TApi::streamDestroy(stream));
}

#endif

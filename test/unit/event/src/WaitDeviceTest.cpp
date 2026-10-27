/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/test/queue/Queue.hpp>
#include <aV2P/test/queue/QueueTestFixture.hpp>
#include <aV2P/wait/Traits.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <memory>

//! A device-wide wait must cover work that was enqueued on a queue of that device.
TEMPLATE_LIST_TEST_CASE("waitDeviceWaitsForEnqueuedWork", "[wait][dev]", aV2P::test::TestQueues)
{
    using Fixture = aV2P::test::QueueTestFixture<TestType>;

    Fixture fixture;

    auto completed = std::make_shared<std::atomic<bool>>(false);
    auto const hostTask = [completed]() { completed->store(true, std::memory_order_release); };

    aV2P::enqueue(fixture.m_queue, hostTask);
    aV2P::wait(fixture.m_dev);

    REQUIRE(completed->load(std::memory_order_acquire));
}

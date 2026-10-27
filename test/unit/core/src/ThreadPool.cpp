/* Copyright 2023 Bernhard Manfred Gruber, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/core/ThreadPool.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("threadpool", "[core]")
{
    aV2P::core::detail::ThreadPool tp{2};

#if AV2P_COMP_CLANG >= AV2P_VERSION_NUMBER(21, 1, 0) && AV2P_COMP_CLANG < AV2P_VERSION_NUMBER(22, 0, 0)
    // This triggers a false positive with clang 21.1.x, including icpx 2025.3.x.
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wmissing-noreturn"
#endif
    auto f1 = tp.enqueueTask([] { throw std::runtime_error("42"); });
    auto f2 = tp.enqueueTask([] { throw 42; });
    auto f3 = tp.enqueueTask([]() noexcept {});
#if AV2P_COMP_CLANG >= AV2P_VERSION_NUMBER(21, 1, 0) && AV2P_COMP_CLANG < AV2P_VERSION_NUMBER(22, 0, 0)
#    pragma clang diagnostic pop
#endif

    CHECK_THROWS_AS(f1.get(), std::runtime_error);

#ifndef AV2P_USES_TSAN
    try
    {
        f2.get();
    }
    catch(int i)
    {
        CHECK(i == 42);
    }
#endif

    CHECK_NOTHROW(f3.get());
}

/* Copyright 2022 Andrea Bocci, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/alpaka.hpp>
#include <aV2P/test/acc/TestAccs.hpp>

#include <catch2/catch_message.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <cstring>
#include <iostream>

// fill an trivial type with std::memset
template<typename T>
constexpr auto memset_value(int c) -> T
{
    T t;
    std::memset(&t, c, sizeof(T));
    return t;
}

// 0- and 1- dimensional space
using Idx = std::size_t;
using Dim1D = aV2P::DimInt<1u>;
using Vec1D = aV2P::Vec<Dim1D, Idx>;

// enabled accelerators with 1-dimensional kernel space
using TestAccs = aV2P::test::EnabledAccs<Dim1D, Idx>;

TEMPLATE_LIST_TEST_CASE("hostOnlyAPI", "[hostOnlyAPI]", TestAccs)
{
    using Device = aV2P::Dev<TestType>;
    using DeviceQueue = aV2P::Queue<Device, aV2P::NonBlocking>;

    auto const platformAcc = aV2P::Platform<TestType>{};

    using Host = aV2P::DevCpu;
    using HostQueue = aV2P::Queue<Host, aV2P::Blocking>;

    // CPU host
    auto const platformHost = aV2P::PlatformCpu{};
    auto const host = aV2P::getDevByIdx(platformHost, 0);
    INFO("using alpaka device: " << aV2P::getName(host));
    HostQueue hostQueue(host);

    // host buffer
    auto h_buffer1 = aV2P::allocMappedBufIfSupported<int, Idx>(host, platformAcc, Vec1D{Idx{42}});
    INFO(
        "host buffer allocated at " << std::data(h_buffer1) << " with " << aV2P::getExtentProduct(h_buffer1)
                                    << " element(s)");

    // async host buffer
    auto h_buffer2 = aV2P::allocAsyncBufIfSupported<int, Idx>(hostQueue, Vec1D{Idx{42}});
    INFO(
        "second host buffer allocated at " << std::data(h_buffer2) << " with " << aV2P::getExtentProduct(h_buffer2)
                                           << " element(s)");

    // host-side memset
    int const value1 = 42;
    int const expected1 = memset_value<int>(value1);
    INFO("host-side memset");
    aV2P::memset(hostQueue, h_buffer1, value1);
    aV2P::wait(hostQueue);
    CHECK(expected1 == *std::data(h_buffer1));

    // host-side async memset
    int const value2 = 99;
    int const expected2 = memset_value<int>(value2);
    INFO("host-side async memset");
    aV2P::memset(hostQueue, h_buffer2, value2);
    aV2P::wait(hostQueue);
    CHECK(expected2 == *std::data(h_buffer2));

    // host-host copies
    INFO("buffer host-host copies");
    aV2P::memcpy(hostQueue, h_buffer2, h_buffer1);
    aV2P::wait(hostQueue);
    CHECK(expected1 == *std::data(h_buffer2));
    aV2P::memcpy(hostQueue, h_buffer1, h_buffer2);
    aV2P::wait(hostQueue);
    CHECK(expected1 == *std::data(h_buffer1));

    // GPU device
    auto const device = aV2P::getDevByIdx(platformAcc, 0);
    INFO("using alpaka device: " << aV2P::getName(device));
    DeviceQueue deviceQueue(device);

    // device buffer
    auto d_buffer1 = aV2P::allocBuf<int, Idx>(device, Vec1D{Idx{42}});
    INFO(
        "device buffer allocated at " << std::data(d_buffer1) << " with " << aV2P::getExtentProduct(d_buffer1)
                                      << " element(s)");

    // async or second sync device buffer
    auto d_buffer2 = aV2P::allocAsyncBufIfSupported<int, Idx>(deviceQueue, Vec1D{Idx{42}});
    INFO(
        "second device buffer allocated at " << std::data(d_buffer2) << " with " << aV2P::getExtentProduct(d_buffer2)
                                             << " element(s)");

    // host-device copies
    INFO("host-device copies");
    aV2P::memcpy(deviceQueue, d_buffer1, h_buffer1);
    aV2P::memcpy(deviceQueue, d_buffer2, h_buffer2);

    // device-device copies
    INFO("device-device copies");
    aV2P::memcpy(deviceQueue, d_buffer1, d_buffer2);
    aV2P::memcpy(deviceQueue, d_buffer2, d_buffer1);

    // device-side memset
    INFO("device-side memset");
    aV2P::memset(deviceQueue, d_buffer1, value1);
    aV2P::memset(deviceQueue, d_buffer2, value2);

    // device-host copies
    INFO("device-host copies");
    aV2P::memcpy(deviceQueue, h_buffer1, d_buffer1);
    aV2P::memcpy(deviceQueue, h_buffer2, d_buffer2);

    aV2P::wait(deviceQueue);
    CHECK(expected1 == *std::data(h_buffer1));
    CHECK(expected2 == *std::data(h_buffer2));
}

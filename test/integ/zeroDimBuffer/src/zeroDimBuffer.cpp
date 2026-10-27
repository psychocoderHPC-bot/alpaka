/* Copyright 2022 Andrea Bocci, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

// make sure the CPU_B_SEQ_T_SEQ backend is always available
#ifndef AV2P_ACC_CPU_B_SEQ_T_SEQ_ENABLED
#define AV2P_ACC_CPU_B_SEQ_T_SEQ_ENABLED
#endif // AV2P_ACC_CPU_B_SEQ_T_SEQ_ENABLED
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
constexpr auto memset_value(std::uint8_t c) -> T
{
    T t;
    std::memset(&t, static_cast<int>(c), sizeof(T));
    return t;
}

// 0- and 1- dimensional space
using Idx = std::size_t;
using Dim0D = aV2P::DimInt<0u>;
using Dim1D = aV2P::DimInt<1u>;
using Scalar = aV2P::Vec<Dim0D, Idx>;

// enabled accelerators with 1-dimensional kernel space
using TestAccs = aV2P::test::EnabledAccs<Dim1D, Idx>;

// aV2P 0D limitation: alpaka 3.0.0 has no zero-dimensional memory support
// (onHost::alloc rejects Vec<T,0> extents via dim()/x() in api/util.hpp:106;
// View requires T_Extents::UniVec, mem/View.hpp:104).
// Restore when alpaka 3 supports 0D Buf/View/alloc; re-enable together with the adapter.
#if 0
TEMPLATE_LIST_TEST_CASE("zeroDimBuffer", "[zeroDimBuffer]", TestAccs)
{
    using DeviceAcc = TestType;
    using DeviceQueue = aV2P::Queue<DeviceAcc, aV2P::NonBlocking>;

    using HostAcc = aV2P::AccCpuSerial<Dim1D, Idx>;
    using HostQueue = aV2P::Queue<HostAcc, aV2P::Blocking>;

    // check that a Scalar extent has exactly 1 element
    Scalar scalar;
    INFO("Scalar size: " << scalar.prod());
    CHECK(scalar.prod() == 1u);

    // CPU host
    auto const platformHost = aV2P::PlatformCpu{};
    auto const host = aV2P::getDevByIdx(platformHost, 0);
    INFO("Using alpaka accelerator: " << aV2P::getAccName<HostAcc>());
    HostQueue hostQueue(host);

    // host buffer
    auto h_buffer1 = aV2P::allocBuf<int, Idx>(host, Scalar{});
    INFO(
        "host buffer allocated at " << std::data(h_buffer1) << " with " << aV2P::getExtentProduct(h_buffer1)
                                    << " element(s)");

    // extents
    CHECK(aV2P::getExtents(h_buffer1) == aV2P::Vec<aV2P::DimInt<0>, Idx>{});
    CHECK(aV2P::getWidth(h_buffer1) == 1);
    CHECK(aV2P::getHeight(h_buffer1) == 1);
    CHECK(aV2P::getDepth(h_buffer1) == 1);

    // async host buffer
    auto h_buffer2 = aV2P::allocAsyncBufIfSupported<int, Idx>(hostQueue, Scalar{});
    INFO(
        "second host buffer allocated at " << std::data(h_buffer2) << " with " << aV2P::getExtentProduct(h_buffer2)
                                           << " element(s)");

    // host-side buffer memset
    std::uint8_t const value1 = 42;
    int const expected1 = memset_value<int>(value1);
    INFO("host-side buffer memset");
    aV2P::memset(hostQueue, h_buffer1, value1);
    aV2P::wait(hostQueue);
    CHECK(expected1 == *h_buffer1);

    // host-side async buffer memset
    std::uint8_t const value2 = 99;
    int const expected2 = memset_value<int>(value2);
    INFO("host-side async buffer memset");
    aV2P::memset(hostQueue, h_buffer2, value2);
    aV2P::wait(hostQueue);
    CHECK(expected2 == *h_buffer2);

    // host-host copies
    INFO("buffer host-host copies");
    aV2P::memcpy(hostQueue, h_buffer2, h_buffer1);
    aV2P::wait(hostQueue);
    CHECK(expected1 == *h_buffer2);
    aV2P::memcpy(hostQueue, h_buffer1, h_buffer2);
    aV2P::wait(hostQueue);
    CHECK(expected1 == *h_buffer1);

    // GPU device
    auto const platformAcc = aV2P::Platform<DeviceAcc>{};
    auto const device = aV2P::getDevByIdx(platformAcc, 0);
    INFO("Using alpaka accelerator: " << aV2P::getAccName<DeviceAcc>());
    DeviceQueue deviceQueue(device);

    // device buffer
    auto d_buffer1 = aV2P::allocBuf<int, Idx>(device, Scalar{});
    INFO(
        "device buffer allocated at " << std::data(d_buffer1) << " with " << aV2P::getExtentProduct(d_buffer1)
                                      << " element(s)");

    // async or second sync device buffer
    auto d_buffer2 = aV2P::allocAsyncBufIfSupported<int, Idx>(deviceQueue, Scalar{});
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

    // device-side buffer memset
    INFO("device-side buffer memset");
    aV2P::memset(deviceQueue, d_buffer1, value1);
    aV2P::memset(deviceQueue, d_buffer2, value2);

    // device-host copies
    INFO("device-host copies");
    aV2P::memcpy(deviceQueue, h_buffer1, d_buffer1);
    aV2P::memcpy(deviceQueue, h_buffer2, d_buffer2);

    aV2P::wait(deviceQueue);
    CHECK(expected1 == *h_buffer1);
    CHECK(expected2 == *h_buffer2);
}
#endif

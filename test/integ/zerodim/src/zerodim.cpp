/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

// Zero-dimensional round trip through aV2P/zerodim: host allocation, host view, 0-D memset,
// host-to-device and device-to-host 0-D memcpy, and the advertised extent product.

#include <aV2P/alpaka.hpp>
#include <aV2P/test/acc/TestAccs.hpp>
#include <aV2P/zerodim/ZeroDim.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <cstdint>
#include <cstring>

using Idx = std::size_t;
using Dim1D = aV2P::DimInt<1u>;
using TestAccs = aV2P::test::EnabledAccs<Dim1D, Idx>;

TEMPLATE_LIST_TEST_CASE("zerodimScalarRoundTrip", "[zerodim]", TestAccs)
{
    using Acc = TestType;
    using Dev = aV2P::Dev<Acc>;
    using Queue = aV2P::Queue<Dev, aV2P::NonBlocking>;

    auto const host = aV2P::getDevByIdx(aV2P::Platform<aV2P::DevCpu>{}, 0u);
    aV2P::Queue<aV2P::DevCpu, aV2P::Blocking> hostQueue{host};

    auto hostBuf = aV2P::zerodim::make_host_buffer<int>();
    auto hostView = aV2P::zerodim::make_host_view(*hostBuf);

    CHECK(aV2P::getExtentProduct(hostBuf) == 1u);
    CHECK(aV2P::getExtentProduct(hostView) == 1u);
    CHECK(aV2P::Dim<decltype(hostBuf)>::value == 0u);

    // 0-D memset covers exactly the single element.
    int expected{};
    std::memset(&expected, 0x2a, sizeof(expected));
    aV2P::memset(hostQueue, hostView, std::uint8_t{0x2a});
    aV2P::wait(hostQueue);
    CHECK(*hostView == expected);

    *hostBuf = 42;

    auto const device = aV2P::getDevByIdx(aV2P::Platform<Acc>{}, 0u);
    Queue deviceQueue{device};
    auto deviceBuf = aV2P::zerodim::make_device_buffer<int>(deviceQueue);
    CHECK(aV2P::getExtentProduct(deviceBuf) == 1u);

    aV2P::memcpy(deviceQueue, deviceBuf, hostView);
    *hostBuf = 0;
    aV2P::memcpy(deviceQueue, hostView, deviceBuf);
    aV2P::wait(deviceQueue);

    CHECK(*hostBuf == 42);
}

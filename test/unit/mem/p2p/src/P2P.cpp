/* Copyright 2023 Axel Hübl, Benjamin Worpitz, Matthias Werner, Bernhard Manfred Gruber, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/mem/buf/Traits.hpp>
#include <aV2P/meta/ForEachType.hpp>
#include <aV2P/test/Extent.hpp>
#include <aV2P/test/acc/TestAccs.hpp>
#include <aV2P/test/mem/view/ViewTest.hpp>
#include <aV2P/test/queue/Queue.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <numeric>
#include <type_traits>

template<typename TAcc>
static auto testP2P(aV2P::Vec<aV2P::Dim<TAcc>, aV2P::Idx<TAcc>> const& extent) -> void
{
    using Dev = aV2P::Dev<TAcc>;
    using Queue = aV2P::test::DefaultQueue<Dev>;

    using Elem = std::uint32_t;
    using Idx = aV2P::Idx<TAcc>;

    auto const platformAcc = aV2P::Platform<TAcc>{};
    if(aV2P::getDevCount(platformAcc) < 2)
    {
        std::cerr << "No two devices found to test peer-to-peer copy." << std::endl;
        CHECK(true);
        return;
    }

    Dev const dev0 = aV2P::getDevByIdx(platformAcc, 0);
    Dev const dev1 = aV2P::getDevByIdx(platformAcc, 1);
    Queue queue0(dev0);
    Queue queue1(dev1);

    auto buf0 = aV2P::allocBuf<Elem, Idx>(dev0, extent);
    auto buf1 = aV2P::allocBuf<Elem, Idx>(dev1, extent);

    // fill each byte with value 42
    auto const byte(static_cast<uint8_t>(42u));
    aV2P::memset(queue1, buf1, byte);
    aV2P::wait(queue1);

    // copy buffer from device 1 into device 0 buffer
    aV2P::memcpy(queue0, buf0, buf1);
    aV2P::wait(queue0);
    // verify buffer on device 0
    aV2P::test::verifyBytesSet<TAcc>(buf0, byte);
}

TEMPLATE_LIST_TEST_CASE("memP2PTest", "[memP2P]", aV2P::test::TestAccs)
{
    using Acc = TestType;
    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;
    testP2P<Acc>(aV2P::test::extentBuf<Dim, Idx>);
}

/* Copyright 2025 Maria Michailidi, Anna Polova, Abdulrahman Al Marzouqi
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/mem/buf/Traits.hpp>
#include <aV2P/mem/view/Traits.hpp>
#include <aV2P/test/Extent.hpp>
#include <aV2P/test/acc/TestAccs.hpp>
#include <aV2P/test/mem/view/ViewTest.hpp>
#include <aV2P/test/queue/Queue.hpp>
#include <catch2/catch_message.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <numeric>
#include <type_traits>

TEMPLATE_LIST_TEST_CASE("memBufFillPrimitiveValuesTest", "[memBuf]", aV2P::test::TestAccs)
{
    using Acc = TestType;
    using Dev = aV2P::Dev<Acc>;
    using Queue = aV2P::test::DefaultQueue<Dev>;
    using Elem = int;
    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;

    auto const platformHost = aV2P::PlatformCpu{};
    auto const devHost = aV2P::getDevByIdx(platformHost, 0);

    auto const platformAcc = aV2P::Platform<Acc>{};
    auto const dev = aV2P::getDevByIdx(platformAcc, 0);

    INFO("Test fill function on device");
    INFO(aV2P::getName(dev));

    Queue queue(dev);

    auto const extent = aV2P::test::extentBuf<Dim, Idx>;

    auto buf = aV2P::allocBuf<Elem, Idx>(dev, extent);

    constexpr Elem fillVal = 42;
    aV2P::fill(queue, buf, fillVal);

    // Copy result to host and check
    auto bufHost = aV2P::allocBuf<Elem, Idx>(devHost, extent);
    aV2P::memcpy(queue, bufHost, buf);
    aV2P::wait(queue);

    Elem const* ptr = std::data(bufHost);
    Idx const size = aV2P::getExtentProduct(bufHost);
    bool passed = true;
    for(Idx i = 0; i < size; ++i)
    {
        if(ptr[i] != fillVal)
        {
            passed = false;
        }
    }
    CHECK(passed);
}

struct Elem
{
    int i;
    float f;

    bool operator==(Elem const& other) const
    {
        return i == other.i && std::fabs(f - other.f) < 1e-9f;
    }
};

TEMPLATE_LIST_TEST_CASE("memBufFillNonPrimitiveValuesTest", "[memBuf]", aV2P::test::TestAccs)
{
    using Acc = TestType;
    using Dev = aV2P::Dev<Acc>;
    using Queue = aV2P::test::DefaultQueue<Dev>;

    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;

    auto const platformHost = aV2P::PlatformCpu{};
    auto const devHost = aV2P::getDevByIdx(platformHost, 0);

    auto const platformAcc = aV2P::Platform<Acc>{};
    auto const dev = aV2P::getDevByIdx(platformAcc, 0);

    INFO("Test fill function on device");
    INFO(aV2P::getName(dev));

    Queue queue(dev);

    auto const extent = aV2P::test::extentBuf<Dim, Idx>;

    auto buf = aV2P::allocBuf<Elem, Idx>(dev, extent);

    constexpr Elem fillVal = {42, 99.0f};
    aV2P::fill(queue, buf, fillVal);

    // Copy result to host and check
    auto bufHost = aV2P::allocBuf<Elem, Idx>(devHost, extent);
    aV2P::memcpy(queue, bufHost, buf);
    aV2P::wait(queue);

    Elem const* ptr = std::data(bufHost);
    Idx const size = aV2P::getExtentProduct(bufHost);
    bool passed = true;
    for(Idx i = 0; i < size; ++i)
    {
        if(ptr[i] != fillVal)
        {
            passed = false;
        }
    }
    CHECK(passed);
}

// aV2P 0D limitation: alpaka 3.0.0 has no zero-dimensional memory support
// (onHost::alloc rejects Vec<T,0> extents via dim()/x() in api/util.hpp:106;
// View requires T_Extents::UniVec, mem/View.hpp:104).
// Restore when alpaka 3 supports 0D Buf/View/alloc; re-enable together with the adapter.
#if 0
TEMPLATE_LIST_TEST_CASE("memBufFillScalarFloatTest", "[memBuf]", aV2P::test::TestAccs)
{
    using Acc = TestType;
    using Dev = aV2P::Dev<Acc>;
    using Queue = aV2P::test::DefaultQueue<Dev>;
    using ElemT = float;
    using Idx = aV2P::Idx<Acc>;

    float epsilon = 1e-9f;

    auto const platformHost = aV2P::PlatformCpu{};
    auto const devHost = aV2P::getDevByIdx(platformHost, 0);

    auto const platformAcc = aV2P::Platform<Acc>{};
    auto const dev = aV2P::getDevByIdx(platformAcc, 0);

    INFO("Test fill function on device");
    INFO(aV2P::getName(dev));

    Queue queue(dev);

    auto const extent = aV2P::test::extentBuf<aV2P::DimInt<0u>, Idx>;

    auto buf = aV2P::allocBuf<ElemT, Idx>(dev, extent);

    constexpr ElemT fillVal = 42.0f;
    aV2P::fill(queue, buf, fillVal);

    // Copy result to host and check
    auto bufHost = aV2P::allocBuf<ElemT, Idx>(devHost, extent);
    aV2P::memcpy(queue, bufHost, buf);
    aV2P::wait(queue);

    ElemT const* ptr = std::data(bufHost);
    CHECK(std::fabs(ptr[0] - fillVal) < epsilon);
}
#endif

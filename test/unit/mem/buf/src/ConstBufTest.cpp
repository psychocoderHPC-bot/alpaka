/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/mem/buf/Traits.hpp>
#include <aV2P/test/Extent.hpp>
#include <aV2P/test/acc/TestAccs.hpp>
#include <aV2P/test/mem/view/ViewTest.hpp>
#include <aV2P/test/queue/Queue.hpp>
#include <catch2/catch_message.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <numeric>
#include <type_traits>

namespace buftest
{
    template<typename TDev, typename TDim, typename TElem, typename TIdx, typename TExtent>
    auto initConstBuf(TDev dev, TExtent extent)
    {
        using QueueAcc = aV2P::test::DefaultQueue<TDev>;

        // allocate buffer
        auto buf = aV2P::allocBuf<TElem, TIdx>(dev, extent);

        // fill buffer
        auto const platformHost = aV2P::PlatformCpu{};
        auto const devHost = aV2P::getDevByIdx(platformHost, 0);
        std::vector<TElem> dataHost(static_cast<std::size_t>(getExtentProduct(extent)), static_cast<TElem>(1));
        auto bufHost = aV2P::createView(devHost, dataHost.data(), extent);
        QueueAcc queueAcc{dev};
        aV2P::memcpy(queueAcc, buf, bufHost);
        aV2P::wait(queueAcc);

#if AV2P_COMP_CLANG >= AV2P_VERSION_NUMBER(21, 1, 0)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wnrvo"
#endif

        // make the buffer constant and return only that
        auto const constBuf = aV2P::makeConstBuf(std::move(buf));
        return constBuf;

#if AV2P_COMP_CLANG >= AV2P_VERSION_NUMBER(21, 1, 0)
#    pragma clang diagnostic pop
#endif
    }

    template<typename TBuf>
    concept onlyConstNativePtr = requires(TBuf t) {
        {
            aV2P::getPtrNative(t)
        } -> std::same_as<aV2P::Elem<std::remove_const_t<TBuf>> const*>;
    };
} // namespace buftest

template<typename TAcc>
static auto testConstBuffer(aV2P::Vec<aV2P::Dim<TAcc>, aV2P::Idx<TAcc>> const& extent) -> void
{
    using Dev = aV2P::Dev<TAcc>;

    using Elem = float;
    using Dim = aV2P::Dim<TAcc>;
    using Idx = aV2P::Idx<TAcc>;

    auto const platformAcc = aV2P::Platform<TAcc>{};
    auto const dev = aV2P::getDevByIdx(platformAcc, 0);

    // aV2P::malloc
    auto buf = aV2P::allocBuf<Elem, Idx>(dev, extent);
    auto const c_buf = buftest::initConstBuf<Dev, Dim, Elem, Idx>(dev, extent);

    using TBuf = decltype(buf);
    using TCBuf = decltype(c_buf);

    auto const offset = aV2P::Vec<Dim, Idx>::zeros();
    aV2P::test::testViewImmutable<Elem const>(c_buf, dev, extent, offset);

    // check that Constant buffer can't be converted to non-const buffer
    STATIC_REQUIRE_FALSE(std::convertible_to<TCBuf, TBuf>);
    STATIC_REQUIRE(std::convertible_to<TCBuf, TCBuf>);
    STATIC_REQUIRE(std::convertible_to<TBuf, TBuf>);
    STATIC_REQUIRE(std::convertible_to<TBuf, TCBuf>);

    STATIC_REQUIRE_FALSE(buftest::onlyConstNativePtr<TBuf>);
    STATIC_REQUIRE(buftest::onlyConstNativePtr<TCBuf>);
    // *getPtrNative(c_buf) = 0.f;  // <- this does not compile, as desired

    // check return types of the buffers
    using Platform = aV2P::Platform<TAcc>;
    if constexpr(std::is_same_v<Platform, aV2P::PlatformCpu> or std::is_same_v<aV2P::AccToTag<TAcc>, aV2P::TagCpuSycl>)
    {
        STATIC_REQUIRE(std::is_same_v<decltype(buf[0]), Elem&>);
        STATIC_REQUIRE(std::is_same_v<decltype(c_buf[0]), Elem const&>);
    }

    // check movability construction of buffers
    STATIC_REQUIRE(std::movable<TBuf>);
    STATIC_REQUIRE(std::movable<std::remove_const_t<TCBuf>>);
}

template<typename TAcc>
static auto testConstBufLifetime(aV2P::Vec<aV2P::Dim<TAcc>, aV2P::Idx<TAcc>> const& extent) -> void
{
    using Dev = aV2P::Dev<TAcc>;
    using Queue = aV2P::test::DefaultQueue<Dev>;

    using Elem = std::uint32_t;
    using Dim = aV2P::Dim<TAcc>;
    using Idx = aV2P::Idx<TAcc>;

    auto const platformAcc = aV2P::Platform<TAcc>{};
    auto const dev = aV2P::getDevByIdx(platformAcc, 0);

    // init and return a const buffer filled with all ones
    auto const c_buf = buftest::initConstBuf<Dev, Dim, Elem, Idx>(dev, extent);

    Queue queue{dev};

    // create local buffer filled with all ones
    auto const platformHost = aV2P::PlatformCpu{};
    auto const devHost = aV2P::getDevByIdx(platformHost, 0);
    std::vector<Elem> dataHost(static_cast<std::size_t>(getExtentProduct(extent)), static_cast<Elem>(1));
    auto bufHost = aV2P::createView(devHost, dataHost.data(), extent);

    auto copiedBack = aV2P::allocBuf<Elem, Idx>(devHost, extent);
    aV2P::memcpy(queue, copiedBack, c_buf);
    aV2P::wait(queue);

    bool resultCorrect = true;
    auto const pHostData = std::data(bufHost);
    auto const pCopiedBackData = std::data(copiedBack);
    for(Idx i(0u); i < getExtentProduct(extent); ++i)
    {
        if(pHostData[i] != pCopiedBackData[i])
        {
            resultCorrect = false;
        }
    }

    REQUIRE(resultCorrect);
}

TEMPLATE_LIST_TEST_CASE("constMemBufBasicTest", "[memBuf]", aV2P::test::TestAccs)
{
    using Acc = TestType;
    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;
    testConstBuffer<Acc>(aV2P::test::extentBuf<Dim, Idx>);
    testConstBufLifetime<Acc>(aV2P::test::extentBuf<Dim, Idx>);
}

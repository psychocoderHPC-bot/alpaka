/* Copyright 2026 Axel Hübl, Benjamin Worpitz, Erik Zenker, Bernhard Manfred Gruber, Jan Stephan, Simone Balducci
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/core/Config.hpp>
#include <aV2P/mem/view/ViewPlainPtr.hpp>
#include <aV2P/meta/ForEachType.hpp>
#include <aV2P/test/Extent.hpp>
#include <aV2P/test/acc/TestAccs.hpp>
#include <aV2P/test/mem/view/ViewTest.hpp>
#include <aV2P/test/queue/Queue.hpp>

#include <catch2/catch_template_test_macros.hpp>

#include <algorithm>
#include <numeric>
#include <tuple>
#include <type_traits>

namespace aV2P::test
{
    template<typename TAcc, typename TElem, bool Const>
    auto testViewPlainPtr() -> void
    {
        using Dev = aV2P::Dev<TAcc>;
        using Dim = aV2P::Dim<TAcc>;
        using Idx = aV2P::Idx<TAcc>;
        using View = aV2P::ViewPlainPtr<Dev, TElem, Dim, Idx>;

        auto const platform = aV2P::Platform<TAcc>{};
        auto const dev = aV2P::getDevByIdx(platform, 0);

        auto const bufExtent = aV2P::test::extentBuf<Dim, Idx>;
        auto buf = aV2P::allocBuf<TElem, Idx>(dev, bufExtent);

        auto const viewExtent = bufExtent;
        auto const viewOffset = aV2P::Vec<Dim, Idx>::all(static_cast<Idx>(0));
        View view(
            aV2P::getPtrNative(buf),
            aV2P::getDev(buf),
            aV2P::getExtents(buf),
            aV2P::getPitchesInBytes(buf));

        aV2P::test::testViewImmutable<TElem>(std::as_const(view), dev, viewExtent, viewOffset);
        if constexpr(!Const)
        {
            using Queue = aV2P::test::DefaultQueue<Dev>;
            Queue queue(dev);
            aV2P::test::testViewMutable<TAcc>(queue, view);
        }
    }

    template<typename TAcc, typename TElem>
    auto testViewPlainPtrOperators() -> void
    {
        using Dev = aV2P::Dev<TAcc>;
        using Dim = aV2P::Dim<TAcc>;
        using Idx = aV2P::Idx<TAcc>;
        using View = aV2P::ViewPlainPtr<Dev, TElem, Dim, Idx>;

        auto const platform = aV2P::Platform<TAcc>{};
        auto const dev = aV2P::getDevByIdx(platform, 0);

        auto buf = aV2P::allocBuf<TElem, Idx>(dev, aV2P::test::extentBuf<Dim, Idx>);
        auto nativePtr = aV2P::getPtrNative(buf);
        View view(nativePtr, aV2P::getDev(buf), aV2P::getExtents(buf), aV2P::getPitchesInBytes(buf));

        // copy-constructor
        View viewCopy(view);
        CHECK(aV2P::getPtrNative(viewCopy) == nativePtr);

        // move-constructor
        View viewMove(std::move(viewCopy));
        CHECK(aV2P::getPtrNative(viewMove) == nativePtr);

        auto buf2 = aV2P::allocBuf<TElem, Idx>(dev, aV2P::test::extentBuf<Dim, Idx>);
        auto nativePtr2 = aV2P::getPtrNative(buf);
        View view2(nativePtr2, aV2P::getDev(buf2), aV2P::getExtents(buf2), aV2P::getPitchesInBytes(buf2));

        // copy-assign
        viewCopy = view2;
        CHECK(aV2P::getPtrNative(viewCopy) == nativePtr2);

        // move-assign
        viewMove = std::move(view2);
        CHECK(aV2P::getPtrNative(viewMove) == nativePtr2);
    }
} // namespace aV2P::test

TEMPLATE_LIST_TEST_CASE("viewPlainPtrTest", "[memView]", aV2P::test::TestAccs)
{
    aV2P::test::testViewPlainPtr<TestType, float, false>();
}

TEMPLATE_LIST_TEST_CASE("viewPlainPtrConstTest", "[memView]", aV2P::test::TestAccs)
{
    aV2P::test::testViewPlainPtr<TestType, float, true>();
}

TEMPLATE_LIST_TEST_CASE("viewPlainPtrOperatorTest", "[memView]", aV2P::test::TestAccs)
{
    aV2P::test::testViewPlainPtrOperators<TestType, float>();
}

// TODO(SimeonEhrig): if C++ 20 is minimum requirement, remove `using`
// replace it with `TEMPLATE_TEST_CASE("Vec generator constructor", "[vec]", std::size_t, int, unsigned, float,
// double)`
// missing feature: parameter of a variadic macro is a C++20 extension
using CreateViewTestTypes = std::tuple<std::array<float, 4>, std::vector<float>>;

TEMPLATE_LIST_TEST_CASE("createView", "[memView]", CreateViewTestTypes)
{
    using Dev = aV2P::DevCpu;
    auto const platform = aV2P::PlatformCpu{};
    auto const dev = aV2P::getDevByIdx(platform, 0);
    aV2P::test::DefaultQueue<Dev> queue(dev);

    TestType a{1, 2, 3, 4};

    // pointer overload
    {
        auto view = aV2P::createView(dev, a.data(), 4);
        STATIC_REQUIRE(std::is_same_v<decltype(view), aV2P::ViewPlainPtr<Dev, float, aV2P::DimInt<1>, int>>);
        STATIC_REQUIRE(aV2P::Dim<decltype(view)>::value == 1);
        CHECK(aV2P::getExtents(view)[0] == 4);

        auto view_from_queue = aV2P::createView(queue, a.data(), 4);
        STATIC_REQUIRE(std::is_same_v<decltype(view), aV2P::ViewPlainPtr<Dev, float, aV2P::DimInt<1>, int>>);
        STATIC_REQUIRE(aV2P::Dim<decltype(view)>::value == 1);
        CHECK(aV2P::getExtents(view)[0] == 4);
    }

    // container and size overload
    {
        auto view = aV2P::createView(dev, a, 4L);
        STATIC_REQUIRE(std::is_same_v<decltype(view), aV2P::ViewPlainPtr<Dev, float, aV2P::DimInt<1>, long>>);
        STATIC_REQUIRE(aV2P::Dim<decltype(view)>::value == 1);
        CHECK(aV2P::getExtents(view)[0] == 4);

        auto view_from_queue = aV2P::createView(queue, a, 4L);
        STATIC_REQUIRE(std::is_same_v<decltype(view), aV2P::ViewPlainPtr<Dev, float, aV2P::DimInt<1>, long>>);
        STATIC_REQUIRE(aV2P::Dim<decltype(view)>::value == 1);
        CHECK(aV2P::getExtents(view)[0] == 4);
    }

    // container overload
    {
        auto view = aV2P::createView(dev, a);
        STATIC_REQUIRE(
            std::is_same_v<decltype(view), aV2P::ViewPlainPtr<Dev, float, aV2P::DimInt<1>, std::size_t>>);
        STATIC_REQUIRE(aV2P::Dim<decltype(view)>::value == 1);
        CHECK(aV2P::getExtents(view)[0] == 4);

        auto view_from_queue = aV2P::createView(queue, a);
        STATIC_REQUIRE(
            std::is_same_v<decltype(view), aV2P::ViewPlainPtr<Dev, float, aV2P::DimInt<1>, std::size_t>>);
        STATIC_REQUIRE(aV2P::Dim<decltype(view)>::value == 1);
        CHECK(aV2P::getExtents(view)[0] == 4);
    }

    decltype(a) b{0, 0, 0, 0};

    // test as temporaries to memcpy
    aV2P::memcpy(queue, aV2P::createView(queue, b, std::size_t{4}), aV2P::createView(queue, a));
    aV2P::wait(queue);
    CHECK(a == b);

    // test as temporaries to memset
    aV2P::memset(queue, aV2P::createView(queue, a), 0);
    aV2P::wait(queue);
    CHECK(a == decltype(a){0, 0, 0, 0});
}

TEMPLATE_LIST_TEST_CASE("createViewSpan", "[memView]", std::tuple<std::span<int>>)
{
    using Dev = aV2P::DevCpu;
    auto const platform = aV2P::PlatformCpu{};
    auto const dev = aV2P::getDevByIdx(platform, 0);

    std::vector vec{1, 2, 3, 4};
#if AV2P_COMP_CLANG
#    pragma clang diagnostic push
#    if __has_warning("-Wunsafe-buffer-usage-in-container")
#        pragma clang diagnostic ignored "-Wunsafe-buffer-usage-in-container"
#    endif
#endif

    std::span<int> a(vec.begin(), vec.end());
#if AV2P_COMP_CLANG
#    pragma clang diagnostic pop
#endif


    // pointer overload
    {
        auto view = aV2P::createView(dev, a.data(), 4);
        STATIC_REQUIRE(std::is_same_v<decltype(view), aV2P::ViewPlainPtr<Dev, int, aV2P::DimInt<1>, int>>);
        STATIC_REQUIRE(aV2P::Dim<decltype(view)>::value == 1);
        CHECK(aV2P::getExtents(view)[0] == 4);
    }

    // container and size overload
    {
        auto view = aV2P::createView(dev, a, 4L);
        STATIC_REQUIRE(std::is_same_v<decltype(view), aV2P::ViewPlainPtr<Dev, int, aV2P::DimInt<1>, long>>);
        STATIC_REQUIRE(aV2P::Dim<decltype(view)>::value == 1);
        CHECK(aV2P::getExtents(view)[0] == 4);
    }

    // container overload
    {
        auto view = aV2P::createView(dev, a);
        STATIC_REQUIRE(std::is_same_v<decltype(view), aV2P::ViewPlainPtr<Dev, int, aV2P::DimInt<1>, std::size_t>>);
        STATIC_REQUIRE(aV2P::Dim<decltype(view)>::value == 1);
        CHECK(aV2P::getExtents(view)[0] == 4);
    }

    aV2P::test::DefaultQueue<Dev> queue(dev);
    std::vector vec2{0, 0, 0, 0};
#if AV2P_COMP_CLANG
#    pragma clang diagnostic push
#    if __has_warning("-Wunsafe-buffer-usage-in-container")
#        pragma clang diagnostic ignored "-Wunsafe-buffer-usage-in-container"
#    endif
#endif
    std::span<int> b(vec2.begin(), vec2.end());
#if AV2P_COMP_CLANG
#    pragma clang diagnostic pop
#endif

    // test as temporaries to memcpy
    aV2P::memcpy(queue, aV2P::createView(queue, b, std::size_t{4}), aV2P::createView(queue, a));
    aV2P::wait(queue);
    CHECK(std::equal(a.begin(), a.end(), b.begin()));

    // test as temporaries to memset
    aV2P::memset(queue, aV2P::createView(queue, a), 0);
    aV2P::wait(queue);
    for(auto x : a)
    {
        CHECK(x == 0);
    }
}

TEMPLATE_LIST_TEST_CASE("createViewConstSpan", "[memView]", std::tuple<std::span<int const>>)
{
    using Dev = aV2P::DevCpu;
    auto const platform = aV2P::PlatformCpu{};
    auto const dev = aV2P::getDevByIdx(platform, 0);

    std::vector vec{1, 2, 3, 4};
#if AV2P_COMP_CLANG
#    pragma clang diagnostic push
#    if __has_warning("-Wunsafe-buffer-usage-in-container")
#        pragma clang diagnostic ignored "-Wunsafe-buffer-usage-in-container"
#    endif
#endif

    std::span<int const> a(vec.begin(), vec.end());
#if AV2P_COMP_CLANG
#    pragma clang diagnostic pop
#endif


    // pointer overload
    {
        auto view = aV2P::createView(dev, a.data(), 4);
        STATIC_REQUIRE(std::is_same_v<decltype(view), aV2P::ViewPlainPtr<Dev, int const, aV2P::DimInt<1>, int>>);
        STATIC_REQUIRE(aV2P::Dim<decltype(view)>::value == 1);
        CHECK(aV2P::getExtents(view)[0] == 4);
    }

    // container and size overload
    {
        auto view = aV2P::createView(dev, a, 4L);
        STATIC_REQUIRE(std::is_same_v<decltype(view), aV2P::ViewPlainPtr<Dev, int const, aV2P::DimInt<1>, long>>);
        STATIC_REQUIRE(aV2P::Dim<decltype(view)>::value == 1);
        CHECK(aV2P::getExtents(view)[0] == 4);
    }

    // container overload
    {
        auto view = aV2P::createView(dev, a);
        STATIC_REQUIRE(
            std::is_same_v<decltype(view), aV2P::ViewPlainPtr<Dev, int const, aV2P::DimInt<1>, std::size_t>>);
        STATIC_REQUIRE(aV2P::Dim<decltype(view)>::value == 1);
        CHECK(aV2P::getExtents(view)[0] == 4);
    }

    aV2P::test::DefaultQueue<Dev> queue(dev);
    std::vector vec2{0, 0, 0, 0};
#if AV2P_COMP_CLANG
#    pragma clang diagnostic push
#    if __has_warning("-Wunsafe-buffer-usage-in-container")
#        pragma clang diagnostic ignored "-Wunsafe-buffer-usage-in-container"
#    endif
#endif
    std::span<int> b(vec2.begin(), vec2.end());
#if AV2P_COMP_CLANG
#    pragma clang diagnostic pop
#endif

    // test as temporaries to memcpy
    aV2P::memcpy(queue, aV2P::createView(queue, b, std::size_t{4}), aV2P::createView(queue, a));
    aV2P::wait(queue);
    CHECK(std::equal(a.begin(), a.end(), b.begin()));
}

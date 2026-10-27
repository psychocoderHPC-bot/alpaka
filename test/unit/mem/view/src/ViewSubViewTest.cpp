/* Copyright 2023 Axel Hübl, Benjamin Worpitz, Erik Zenker, Bernhard Manfred Gruber, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/core/Config.hpp>
#include <aV2P/mem/view/Traits.hpp>
#include <aV2P/mem/view/ViewSubView.hpp>
#include <aV2P/test/Extent.hpp>
#include <aV2P/test/acc/TestAccs.hpp>
#include <aV2P/test/mem/view/ViewTest.hpp>
#include <aV2P/test/queue/Queue.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <numeric>
#include <type_traits>

namespace aV2P::test
{
    template<typename TAcc, typename TDev, typename TElem, typename TDim, typename TIdx, typename TBuf>
    auto testViewSubViewImmutable(
        aV2P::ViewSubView<TDev, TElem, TDim, TIdx> const& view,
        TBuf& buf,
        TDev const& dev,
        aV2P::Vec<TDim, TIdx> const& extentView,
        aV2P::Vec<TDim, TIdx> const& offsetView) -> void
    {
        aV2P::test::testViewImmutable<TElem>(view, dev, extentView, offsetView);

        // aV2P::trait::GetPitchesInBytes
        // The pitch of the view has to be identical to the pitch of the underlying buffer in all dimensions.
        auto const pitchBuf = aV2P::getPitchesInBytes(buf);
        {
            auto const pitchView = aV2P::getPitchesInBytes(view);
            CHECK(pitchBuf == pitchView);
        }

        // aV2P::trait::GetPtrNative
        // The native pointer has to be exactly the value we calculate here.
        {
            auto viewPtrNative = reinterpret_cast<std::uint8_t*>(aV2P::getPtrNative(buf));
            if constexpr(TDim::value > 0)
                viewPtrNative += (offsetView * pitchBuf).sum();
            REQUIRE(viewPtrNative == reinterpret_cast<std::uint8_t const*>(aV2P::getPtrNative(view)));
        }
    }

    template<typename TAcc, typename TDev, typename TElem, typename TDim, typename TIdx, typename TBuf>
    auto testViewSubViewMutable(
        aV2P::ViewSubView<TDev, TElem, TDim, TIdx>& view,
        TBuf& buf,
        TDev const& dev,
        aV2P::Vec<TDim, TIdx> const& extentView,
        aV2P::Vec<TDim, TIdx> const& offsetView) -> void
    {
        testViewSubViewImmutable<TAcc>(view, buf, dev, extentView, offsetView);

        using Queue = aV2P::test::DefaultQueue<TDev>;
        Queue queue(dev);
        aV2P::test::testViewMutable<TAcc>(queue, view);
    }

    template<typename TAcc, typename TElem>
    auto testViewSubViewNoOffset() -> void
    {
        using Dev = aV2P::Dev<TAcc>;
        using Dim = aV2P::Dim<TAcc>;
        using Idx = aV2P::Idx<TAcc>;
        using View = aV2P::ViewSubView<Dev, TElem, Dim, Idx>;

        auto const platform = aV2P::Platform<TAcc>{};
        auto const dev = aV2P::getDevByIdx(platform, 0);

        auto const bufExtent = aV2P::test::extentBuf<Dim, Idx>;
        auto buf = aV2P::allocBuf<TElem, Idx>(dev, bufExtent);

        auto const viewExtent = bufExtent;
        auto const viewOffset = aV2P::Vec<Dim, Idx>::zeros();
        View view(buf);

        aV2P::test::testViewSubViewMutable<TAcc>(view, buf, dev, viewExtent, viewOffset);
    }

    template<typename TAcc, typename TElem>
    auto testViewSubViewOffset() -> void
    {
        using Dev = aV2P::Dev<TAcc>;
        using Dim = aV2P::Dim<TAcc>;
        using Idx = aV2P::Idx<TAcc>;
        using View = aV2P::ViewSubView<Dev, TElem, Dim, Idx>;

        auto const platform = aV2P::Platform<TAcc>{};
        auto const dev = aV2P::getDevByIdx(platform, 0);

        auto const bufExtent = aV2P::test::extentBuf<Dim, Idx>;
        auto buf = aV2P::allocBuf<TElem, Idx>(dev, bufExtent);

        auto const viewExtent = aV2P::test::extentSubView<Dim, Idx>;
        auto const viewOffset = aV2P::test::offset<Dim, Idx>;
        View view(buf, viewExtent, viewOffset);

        aV2P::test::testViewSubViewMutable<TAcc>(view, buf, dev, viewExtent, viewOffset);
    }

    template<typename TAcc, typename TElem>
    auto testViewSubViewOffsetConst() -> void
    {
        using Dev = aV2P::Dev<TAcc>;
        using Dim = aV2P::Dim<TAcc>;
        using Idx = aV2P::Idx<TAcc>;
        using View = aV2P::ViewSubView<Dev, TElem, Dim, Idx>;

        auto const platform = aV2P::Platform<TAcc>{};
        auto const dev = aV2P::getDevByIdx(platform, 0);

        auto const bufExtent = aV2P::test::extentBuf<Dim, Idx>;
        auto buf = aV2P::allocBuf<TElem, Idx>(dev, bufExtent);

        auto const viewExtent = aV2P::test::extentSubView<Dim, Idx>;
        auto const viewOffset = aV2P::test::offset<Dim, Idx>;
        View const view(buf, viewExtent, viewOffset);

        aV2P::test::testViewSubViewImmutable<TAcc>(view, buf, dev, viewExtent, viewOffset);
    }
} // namespace aV2P::test

TEMPLATE_LIST_TEST_CASE("viewSubViewNoOffsetTest", "[memView]", aV2P::test::TestAccs)
{
    aV2P::test::testViewSubViewNoOffset<TestType, float>();
}

TEMPLATE_LIST_TEST_CASE("viewSubViewOffsetTest", "[memView]", aV2P::test::TestAccs)
{
    aV2P::test::testViewSubViewOffset<TestType, float>();
}

TEMPLATE_LIST_TEST_CASE("viewSubViewOffsetConstTest", "[memView]", aV2P::test::TestAccs)
{
    aV2P::test::testViewSubViewOffsetConst<TestType, float>();
}

TEST_CASE("viewSubViewExample", "[memView]")
{
    using Dev = aV2P::DevCpu;
    using Dim = aV2P::DimInt<2>;
    using Idx = int;
    using Vec = aV2P::Vec<Dim, Idx>;

    auto const platform = aV2P::PlatformCpu{};
    auto const dev = aV2P::getDevByIdx(platform, 0);

    auto const extent = Vec{4, 5};
    auto buf = aV2P::allocBuf<int, Idx>(dev, extent);

    auto checkBufContent = [&](std::vector<std::vector<int>> const& data)
    {
        for(std::size_t row = 0; row < 4; row++)
            for(std::size_t col = 0; col < 5; col++)
            {
                CAPTURE(row, col);
                CHECK(buf[Vec{static_cast<Idx>(row), static_cast<Idx>(col)}] == data[row][col]);
            }
    };

    for(Idx i = 0; i < 4; i++)
        for(Idx j = 0; j < 5; j++)
            buf.at(Vec{i, j}) = 1;
    checkBufContent({{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}});

    {
        // write a row
        auto subView = aV2P::ViewSubView<Dev, int, Dim, Idx>{buf, Vec{1, 5}, Vec{2, 0}};
        for(Idx i = 0; i < 5; i++)
            subView.at(Vec{0, i}) = 2;
    }
    checkBufContent({{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {2, 2, 2, 2, 2}, {1, 1, 1, 1, 1}});

    {
        // write part of a column
        auto subView = aV2P::ViewSubView<Dev, int, Dim, Idx>{buf, Vec{3, 1}, Vec{0, 2}};
        for(Idx i = 0; i < 3; i++)
            subView.at(Vec{i, 0}) = 3;
    }
    checkBufContent({{1, 1, 3, 1, 1}, {1, 1, 3, 1, 1}, {2, 2, 3, 2, 2}, {1, 1, 1, 1, 1}});

    {
        // write the center without a 1-elem border
        auto subView = aV2P::ViewSubView<Dev, int, Dim, Idx>{buf, Vec{2, 3}, Vec{1, 1}};
        for(Idx i = 0; i < 2; i++)
            for(Idx j = 0; j < 3; j++)
                subView.at(Vec{i, j}) = 4;
        checkBufContent({{1, 1, 3, 1, 1}, {1, 4, 4, 4, 1}, {2, 4, 4, 4, 2}, {1, 1, 1, 1, 1}});

        {
            // write the first column of the center region
            auto subSubView = aV2P::ViewSubView<Dev, int, Dim, Idx>{subView, Vec{2, 1}, Vec{0, 0}};
            for(Idx i = 0; i < 2; i++)
                subSubView.at(Vec{i, 0}) = 5;
        }
        checkBufContent({{1, 1, 3, 1, 1}, {1, 5, 4, 4, 1}, {2, 5, 4, 4, 2}, {1, 1, 1, 1, 1}});

        {
            // write the second row of the center region, skipping the first
            auto subSubView = aV2P::ViewSubView<Dev, int, Dim, Idx>{subView, Vec{1, 2}, Vec{1, 1}};
            for(Idx i = 0; i < 2; i++)
                subSubView.at(Vec{0, i}) = 6;
        }
        checkBufContent({{1, 1, 3, 1, 1}, {1, 5, 4, 4, 1}, {2, 5, 6, 6, 2}, {1, 1, 1, 1, 1}});
    }
}

TEST_CASE("calculatePitchesFromExtents", "[memView]")
{
    CHECK((aV2P::detail::calculatePitchesFromExtents<float>(aV2P::Vec{1, 1, 1}) == aV2P::Vec{4, 4, 4}));
    CHECK((aV2P::detail::calculatePitchesFromExtents<float>(aV2P::Vec{2, 2, 2}) == aV2P::Vec{16, 8, 4}));
    CHECK((aV2P::detail::calculatePitchesFromExtents<float>(aV2P::Vec{42, 10, 2}) == aV2P::Vec{80, 8, 4}));
}

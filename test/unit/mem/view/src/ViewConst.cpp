/* Copyright 2022 Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/acc/Tag.hpp>
#include <aV2P/mem/view/ViewConst.hpp>
#include <aV2P/platform/Traits.hpp>
#include <aV2P/test/Extent.hpp>
#include <aV2P/test/acc/TestAccs.hpp>
#include <aV2P/test/mem/view/ViewTest.hpp>
#include <aV2P/test/queue/Queue.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <numeric>
#include <type_traits>

namespace aV2P::test
{
    template<typename TAcc, typename TQualifiedView, typename TDev, typename TQueue, typename TDim, typename TIdx>
    void testViewConst(
        TQualifiedView& view,
        TDev const& dev,
        TQueue& queue,
        aV2P::Vec<TDim, TIdx> const& extents,
        aV2P::Vec<TDim, TIdx> const& offsets)
    {
        using TView = std::remove_const_t<TQualifiedView>;
        STATIC_REQUIRE(std::is_same_v<aV2P::Elem<TView>, float const>);
        STATIC_REQUIRE(std::is_same_v<decltype(aV2P::getPtrNative(view)), float const*>);

        aV2P::test::testViewImmutable<float const>(view, dev, extents, offsets);

        // test copying from view
        auto dstBuf = allocBuf<float, TIdx>(dev, getExtents(view));
        memcpy(queue, dstBuf, view);
        wait(queue);
        verifyViewsEqual<TAcc>(dstBuf, view);

        // test special member functions
        TView viewCopy(view);
        TView viewMove(std::move(viewCopy));
        viewCopy = viewMove;
        viewCopy = std::move(viewMove);

        // test view accessors
        STATIC_REQUIRE(std::is_same_v<decltype(view.data()), float const*>);
        using Platform = aV2P::Platform<TAcc>;
        if constexpr(
            std::is_same_v<Platform, aV2P::PlatformCpu> or std::is_same_v<aV2P::AccToTag<TAcc>, aV2P::TagCpuSycl>)
        {
            if constexpr(TDim::value == 0)
            {
                STATIC_REQUIRE(std::is_same_v<decltype(*view), float const&>);
            }
            else if constexpr(TDim::value == 1)
            {
                STATIC_REQUIRE(std::is_same_v<decltype(view[0]), float const&>);
            }

            STATIC_REQUIRE(std::is_same_v<decltype(view[aV2P::Vec<TDim, TIdx>::zeros()]), float const&>);
            STATIC_REQUIRE(std::is_same_v<decltype(view.at(aV2P::Vec<TDim, TIdx>::zeros())), float const&>);
        }
    }
} // namespace aV2P::test

TEMPLATE_LIST_TEST_CASE("viewConstTest", "[memView]", aV2P::test::TestAccs)
{
    using Elem = float;
    using Acc = TestType;
    using Dev = aV2P::Dev<Acc>;
    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;

    auto const platformAcc = aV2P::Platform<Acc>{};
    auto const dev = aV2P::getDevByIdx(platformAcc, 0);

    auto const extents = aV2P::test::extentBuf<Dim, Idx>;
    auto buf = aV2P::allocBuf<Elem, Idx>(dev, extents);
    auto queue = aV2P::test::DefaultQueue<Dev>{dev};
    aV2P::test::iotaFillView(queue, buf);
    auto const offsets = aV2P::Vec<Dim, Idx>::all(static_cast<Idx>(0));

    auto view = aV2P::ViewConst(buf);
    aV2P::test::testViewConst<Acc>(view, dev, queue, extents, offsets);

    auto const cview = aV2P::ViewConst(buf);
    aV2P::test::testViewConst<Acc>(cview, dev, queue, extents, offsets);

    auto view_cbuf = aV2P::ViewConst(std::as_const(buf));
    aV2P::test::testViewConst<Acc>(view_cbuf, dev, queue, extents, offsets);

    auto const cview_cbuf = aV2P::ViewConst(std::as_const(buf));
    aV2P::test::testViewConst<Acc>(cview_cbuf, dev, queue, extents, offsets);

    auto yolo = aV2P::ViewConst(aV2P::ViewConst(aV2P::ViewConst(cview_cbuf)));
    aV2P::test::testViewConst<Acc>(yolo, dev, queue, extents, offsets);
}

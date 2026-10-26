/* Copyright 2022 Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#ifdef AV2P_USE_MDSPAN
#    include <aV2P/alpaka.hpp>
#    include <aV2P/test/Extent.hpp>
#    include <aV2P/test/acc/TestAccs.hpp>
#    include <aV2P/test/mem/view/ViewTest.hpp>

#    include <catch2/catch_template_test_macros.hpp>

#    if defined(__NVCC__) && !defined(__CUDACC_EXTENDED_LAMBDA__)
#        error "mdspan requires nvcc's extended lambdas"
#    endif

namespace
{
    template<std::size_t... Is>
    constexpr auto make_reverse_index_sequence_impl(std::index_sequence<Is...>)
    {
        return std::index_sequence<(sizeof...(Is) - Is - 1)...>{};
    }

    template<std::size_t N>
    using make_reverse_index_sequence = decltype(make_reverse_index_sequence_impl(std::make_index_sequence<N>{}));
} // namespace

#    if AV2P_COMP_NVCC
#        define NOEXCEPT_UNLESS_NVCC
#    else
#        define NOEXCEPT_UNLESS_NVCC noexcept
#    endif

namespace
{
    //! Checks that the mdspan elements are visited in linearized order.
    template<typename TAcc, typename TSpan, typename TExtent>
    struct MdSpanLoopKernel
    {
        TSpan mds;
        TExtent extent;

        AV2P_FN_ACC void operator()(TAcc const&, bool* success) const NOEXCEPT_UNLESS_NVCC
        {
            auto counter = 0;
            aV2P::meta::ndLoopIncIdx(
                extent,
                [&](auto ind)
                {
                    [[maybe_unused]] auto const a = aV2P::toArray(ind);
                    AV2P_CHECK(*success, mds(a) == counter);
                    counter++;
                });
        }
    };

    //! Checks the elements of a transposed mdspan in reverse index order.
    template<typename TAcc, typename TSpan, typename TExtent>
    struct MdSpanTransposedLoopKernel
    {
        TSpan mds;
        TExtent extent;

        AV2P_FN_ACC void operator()(TAcc const&, bool* success) const NOEXCEPT_UNLESS_NVCC
        {
            auto counter = 0;
            aV2P::meta::ndLoop(
                make_reverse_index_sequence<aV2P::Dim<TAcc>::value>{},
                extent,
                [&](auto ind)
                {
                    [[maybe_unused]] auto const a = aV2P::toArray(ind);
                    AV2P_CHECK(*success, mds(a) == counter);
                    counter++;
                });
        }
    };

    //! Checks submdspan access on a two-dimensional mdspan.
    template<typename TAcc, typename TSpan>
    struct SubMdSpanKernel
    {
        TSpan mds;

        AV2P_FN_ACC void operator()(TAcc const&, bool* success) const NOEXCEPT_UNLESS_NVCC
        {
            AV2P_CHECK(*success, mds(0, 0) == 0);
            AV2P_CHECK(*success, mds(0, 1) == 1);
            AV2P_CHECK(*success, mds(0, 2) == 2);
            AV2P_CHECK(*success, mds(1, 0) == 3);
            AV2P_CHECK(*success, mds(1, 1) == 4);
            AV2P_CHECK(*success, mds(1, 2) == 5);

            auto elem_1_2 = aV2P::experimental::submdspan(mds, 1, 2);
            AV2P_CHECK(*success, elem_1_2() == 5);

            auto row0 = aV2P::experimental::submdspan(mds, 0, aV2P::experimental::full_extent);
            AV2P_CHECK(*success, row0(0) == 0);
            AV2P_CHECK(*success, row0(1) == 1);
            AV2P_CHECK(*success, row0(2) == 2);

            auto row1 = aV2P::experimental::submdspan(mds, 1, aV2P::experimental::full_extent);
            AV2P_CHECK(*success, row1(0) == 3);
            AV2P_CHECK(*success, row1(1) == 4);
            AV2P_CHECK(*success, row1(2) == 5);

            auto col0 = aV2P::experimental::submdspan(mds, aV2P::experimental::full_extent, 0);
            AV2P_CHECK(*success, col0(0) == 0);
            AV2P_CHECK(*success, col0(1) == 3);

            auto col1 = aV2P::experimental::submdspan(mds, aV2P::experimental::full_extent, 1);
            AV2P_CHECK(*success, col1(0) == 1);
            AV2P_CHECK(*success, col1(1) == 4);

            auto col2 = aV2P::experimental::submdspan(mds, aV2P::experimental::full_extent, 2);
            AV2P_CHECK(*success, col2(0) == 2);
            AV2P_CHECK(*success, col2(1) == 5);

            auto matrix = aV2P::experimental::submdspan(
                mds,
                aV2P::experimental::full_extent,
                aV2P::experimental::full_extent);
            AV2P_CHECK(*success, matrix(0, 0) == 0);
            AV2P_CHECK(*success, matrix(0, 1) == 1);
            AV2P_CHECK(*success, matrix(0, 2) == 2);
            AV2P_CHECK(*success, matrix(1, 0) == 3);
            AV2P_CHECK(*success, matrix(1, 1) == 4);
            AV2P_CHECK(*success, matrix(1, 2) == 5);
        }
    };
} // namespace

TEMPLATE_LIST_TEST_CASE("mdSpan", "[memView]", aV2P::test::TestAccs)
{
    using TElem = int;

    using Acc = TestType;
    using Dev = aV2P::Dev<Acc>;
    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;
    using Vec = aV2P::Vec<Dim, Idx>;

    auto const platform = aV2P::Platform<Dev>{};
    auto const dev = aV2P::getDevByIdx(platform, 0);
    auto queue = aV2P::Queue<Dev, aV2P::Blocking>(dev);

    auto const extent = aV2P::test::extentBuf<Dim, Idx>;
    auto buf = aV2P::allocBuf<TElem, Idx>(dev, extent);
    aV2P::test::iotaFillView(queue, buf);

    auto const mds = aV2P::experimental::getMdSpan(buf);
    aV2P::test::KernelExecutionFixture<Acc> fixture(Vec::ones());
    using MdSpanType = decltype(aV2P::experimental::getMdSpan(buf));
    CHECK(fixture(MdSpanLoopKernel<Acc, MdSpanType, Vec>{mds, extent}));

    auto extentTransposed = extent;
    std::reverse(extentTransposed.begin(), extentTransposed.end());
    auto const mdst = aV2P::experimental::getMdSpanTransposed(buf);
    using MdSpanTransposedType = decltype(aV2P::experimental::getMdSpanTransposed(buf));
    CHECK(fixture(MdSpanTransposedLoopKernel<Acc, MdSpanTransposedType, Vec>{mdst, extentTransposed}));
}

TEMPLATE_LIST_TEST_CASE("submdspan", "[memView]", aV2P::test::TestAccs)
{
    using TElem = int;

    using Acc = TestType;
    using Dev = aV2P::Dev<Acc>;
    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;
    using Vec = aV2P::Vec<Dim, Idx>;

    auto const platform = aV2P::Platform<Dev>{};
    auto const dev = aV2P::getDevByIdx(platform, 0);
    auto queue = aV2P::Queue<Dev, aV2P::Blocking>(dev);

    auto const extent = aV2P::Vec{2, 3}; // 2 rows, 3 cols
    auto buf = aV2P::allocBuf<TElem, int>(dev, extent);
    aV2P::test::iotaFillView(queue, buf);

    auto const mds = aV2P::experimental::getMdSpan(buf);
    aV2P::test::KernelExecutionFixture<Acc> fixture(Vec::ones());
    using MdSpanType = decltype(aV2P::experimental::getMdSpan(buf));
    CHECK(fixture(SubMdSpanKernel<Acc, MdSpanType>{mds}));
}
#endif

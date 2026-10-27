/* Copyright 2022 Jeffrey Kelling, Bernhard Manfred Gruber, Jan Stephan, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/dev/Traits.hpp>
#include <aV2P/idx/Accessors.hpp>
#include <aV2P/idx/MapIdx.hpp>
#include <aV2P/mem/view/ViewPlainPtr.hpp>
#include <aV2P/mem/view/ViewSubView.hpp>
#include <aV2P/test/Extent.hpp>
#include <aV2P/test/dim/TestDims.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

template<typename TDim, aV2P::concepts::Tag TAccTag>
auto mapIdxPitchBytes(TAccTag const&)
{
    using Dim = TDim;
    using Idx = std::size_t;
    using Vec = aV2P::Vec<Dim, Idx>;

    auto const extentNd = aV2P::test::extentBuf<Dim, Idx>;

    using Acc = aV2P::TagToAcc<TAccTag, Dim, Idx>;
    using Elem = std::uint8_t;
    auto const platformAcc = aV2P::Platform<Acc>{};
    auto const devAcc = aV2P::getDevByIdx(platformAcc, 0);
    auto parentView = aV2P::createView(devAcc, static_cast<Elem*>(nullptr), extentNd);

    auto const offset = Vec::all(4u);
    auto const extent = Vec::all(4u);
    auto const idxNd = Vec::all(2u);
    auto view = aV2P::createSubView(parentView, extent, offset);
    auto pitch = aV2P::getPitchesInBytes(view);

    auto const idx1d = aV2P::mapIdxPitchBytes<1u>(idxNd, pitch);
    auto const idx1dDelta = aV2P::mapIdx<1u>(idxNd + offset, extentNd) - aV2P::mapIdx<1u>(offset, extentNd);

    auto const idxNdResult = aV2P::mapIdxPitchBytes<Dim::value>(idx1d, pitch);

    // linear index in pitched offset box should be the difference between
    // linear index in parent box and linear index of offset
    REQUIRE(idx1d == idx1dDelta);
    // roundtrip
    REQUIRE(idxNd == idxNdResult);
}

TEMPLATE_LIST_TEST_CASE("mapIdxPitchBytes", "[idx]", aV2P::test::NonZeroTestDims)
{
    // execute the example once for each enabled accelerator
    std::apply([](auto const&... tags) { (mapIdxPitchBytes<TestType>(tags), ...); }, aV2P::EnabledAccTags{});
}

/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera, Bernhard Manfred Gruber, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/idx/Accessors.hpp>
#include <aV2P/idx/MapIdx.hpp>
#include <aV2P/meta/ForEachType.hpp>
#include <aV2P/test/Extent.hpp>
#include <aV2P/test/dim/TestDims.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

TEMPLATE_LIST_TEST_CASE("mapIdx", "[idx]", aV2P::test::TestDims)
{
    using Dim = TestType;
    using Idx = std::size_t;
    using Vec = aV2P::Vec<Dim, Idx>;

    auto const extentNd = aV2P::test::extentBuf<Dim, Idx>;
    auto const idxNd = extentNd - Vec::all(4u);

    auto const idx1d = aV2P::mapIdx<1u>(idxNd, extentNd);

    auto const idxNdResult = aV2P::mapIdx<Dim::value>(idx1d, extentNd);

    REQUIRE(idxNd == idxNdResult);
}

/* Copyright 2026 Sergei Bastrakov, Bernhard Manfred Gruber, Jan Stephan, Andrea Bocci, Simone Balducci
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/dev/Traits.hpp>
#include <aV2P/test/acc/TestAccs.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstddef>

TEMPLATE_LIST_TEST_CASE("getWarpSizes", "[dev]", aV2P::test::TestAccs)
{
    auto const platform = aV2P::Platform<TestType>{};
    auto const dev = aV2P::getDevByIdx(platform, 0);
    auto const warpExtents = aV2P::getWarpSizes(dev);
    REQUIRE(std::all_of(
        std::cbegin(warpExtents),
        std::cend(warpExtents),
        [](std::size_t warpExtent) { return warpExtent > 0; }));
}

TEMPLATE_LIST_TEST_CASE("getPreferredWarpSize", "[dev]", aV2P::test::TestAccs)
{
    auto const platform = aV2P::Platform<TestType>{};
    auto const dev = aV2P::getDevByIdx(platform, 0);
    auto const preferredWarpSize = aV2P::getPreferredWarpSize(dev);
    REQUIRE(preferredWarpSize > 0);
}

TEMPLATE_LIST_TEST_CASE("concepts::Device", "[dev]", aV2P::test::TestAccs)
{
    auto const platform = aV2P::Platform<TestType>{};
    auto const dev = aV2P::getDevByIdx(platform, 0);
    REQUIRE(aV2P::concepts::Device<decltype(dev)>);
}

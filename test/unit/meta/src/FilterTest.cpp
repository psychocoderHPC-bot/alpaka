/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/meta/Filter.hpp>
#include <catch2/catch_test_macros.hpp>

#include <tuple>
#include <type_traits>

TEST_CASE("filter", "[meta]")
{
    using FilterInput = std::tuple<int, float, long>;

    using FilterResult = aV2P::meta::Filter<FilterInput, std::is_integral>;

    using FilterReference = std::tuple<int, long>;

    static_assert(std::is_same_v<FilterReference, FilterResult>, "aV2P::meta::Filter failed!");
}

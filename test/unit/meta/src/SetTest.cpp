/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/meta/Set.hpp>
#include <catch2/catch_test_macros.hpp>

#include <tuple>
#include <type_traits>

TEST_CASE("isSetTrue", "[meta]")
{
    using IsSetInput = std::tuple<int, float, long>;

    constexpr bool IsSetResult = aV2P::meta::IsSet<IsSetInput>::value;

    constexpr bool IsSetReference = true;

    static_assert(IsSetReference == IsSetResult, "aV2P::meta::IsSet failed!");
}

TEST_CASE("isSetFalse", "[meta]")
{
    using IsSetInput = std::tuple<int, float, int>;

    constexpr bool IsSetResult = aV2P::meta::IsSet<IsSetInput>::value;

    constexpr bool IsSetReference = false;

    static_assert(IsSetReference == IsSetResult, "aV2P::meta::IsSet failed!");
}

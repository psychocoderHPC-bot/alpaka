/* Copyright 2022 Jiří Vyskočil, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/meta/IsArrayOrVector.hpp>
#include <aV2P/vec/Vec.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <string>
#include <vector>

TEST_CASE("isArrayOrVector", "[meta]")
{
    STATIC_REQUIRE(aV2P::meta::IsArrayOrVector<std::array<int, 10>>::value);
    STATIC_REQUIRE(aV2P::meta::IsArrayOrVector<std::vector<float>>::value);
    STATIC_REQUIRE(aV2P::meta::IsArrayOrVector<aV2P::Vec<aV2P::DimInt<6u>, float>>::value);

    [[maybe_unused]] float arrayFloat[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    STATIC_REQUIRE(aV2P::meta::IsArrayOrVector<decltype(arrayFloat)>::value);
}

TEST_CASE("isActuallyNotArrayOrVector", "[meta]")
{
    float notAnArrayFloat = 15.0f;
    STATIC_REQUIRE_FALSE(aV2P::meta::IsArrayOrVector<decltype(notAnArrayFloat)>::value);

    [[maybe_unused]] float* notAnArrayFloatPointer = &notAnArrayFloat;
    STATIC_REQUIRE_FALSE(aV2P::meta::IsArrayOrVector<decltype(notAnArrayFloatPointer)>::value);

    std::string notAnArrayString{"alpaka"};
    STATIC_REQUIRE_FALSE(aV2P::meta::IsArrayOrVector<decltype(notAnArrayString)>::value);
}

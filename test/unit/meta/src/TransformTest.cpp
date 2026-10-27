/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/meta/Transform.hpp>
#include <catch2/catch_test_macros.hpp>

#include <tuple>
#include <type_traits>

template<typename T>
using AddConst = T const;

TEST_CASE("transform", "[meta]")
{
    using TransformInput = std::tuple<int, float, long>;

    using TransformResult = aV2P::meta::Transform<TransformInput, AddConst>;

    using TransformReference = std::tuple<int const, float const, long const>;

    static_assert(std::is_same_v<TransformReference, TransformResult>, "aV2P::meta::Transform failed!");
}

TEST_CASE("transformVariadic", "[meta]")
{
    using TransformInput = std::tuple<int, float, long>;

    using TransformResult = aV2P::meta::Transform<TransformInput, std::tuple>;

    using TransformReference = std::tuple<std::tuple<int>, std::tuple<float>, std::tuple<long>>;

    static_assert(std::is_same_v<TransformReference, TransformResult>, "aV2P::meta::Transform failed!");
}

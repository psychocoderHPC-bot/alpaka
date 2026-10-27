/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Bernhard Manfred Gruber, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/dim/DimIntegralConst.hpp>
#include <aV2P/meta/CartesianProduct.hpp>
#include <catch2/catch_test_macros.hpp>

#include <tuple>
#include <type_traits>

TEST_CASE("cartesianProduct", "[meta]")
{
    using TestDims = std::tuple<aV2P::DimInt<1u>, aV2P::DimInt<2u>, aV2P::DimInt<3u>>;

    using TestIdxs = std::tuple<std::size_t, std::int64_t>;

    using CartesianProductResult = aV2P::meta::CartesianProduct<std::tuple, TestDims, TestIdxs>;

    using CartesianProductReference = std::tuple<
        std::tuple<aV2P::DimInt<1u>, std::size_t>,
        std::tuple<aV2P::DimInt<2u>, std::size_t>,
        std::tuple<aV2P::DimInt<3u>, std::size_t>,
        std::tuple<aV2P::DimInt<1u>, std::int64_t>,
        std::tuple<aV2P::DimInt<2u>, std::int64_t>,
        std::tuple<aV2P::DimInt<3u>, std::int64_t>>;

    static_assert(
        std::is_same_v<CartesianProductReference, CartesianProductResult>,
        "aV2P::meta::CartesianProduct failed!");
}

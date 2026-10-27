/* Copyright 2022 Bernhard Manfred Gruber, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/meta/TypeListOps.hpp>
#include <catch2/catch_test_macros.hpp>

#include <tuple>
#include <type_traits>

template<typename... TTypes>
struct TypeList
{
};

TEST_CASE("front", "[meta]")
{
    STATIC_REQUIRE(std::is_same_v<aV2P::meta::Front<std::tuple<int>>, int>);
    STATIC_REQUIRE(std::is_same_v<aV2P::meta::Front<std::tuple<int, int>>, int>);
    STATIC_REQUIRE(std::is_same_v<aV2P::meta::Front<std::tuple<float, int>>, float>);
    STATIC_REQUIRE(std::is_same_v<aV2P::meta::Front<std::tuple<short, int, double, float, float>>, short>);
}

TEST_CASE("contains", "[meta]")
{
    STATIC_REQUIRE(!aV2P::meta::Contains<std::tuple<>, int>::value);
    STATIC_REQUIRE(aV2P::meta::Contains<std::tuple<int>, int>::value);
    STATIC_REQUIRE(aV2P::meta::Contains<std::tuple<short, int, double, float>, short>::value);
    STATIC_REQUIRE(aV2P::meta::Contains<std::tuple<short, int, double, float>, double>::value);
    STATIC_REQUIRE(aV2P::meta::Contains<std::tuple<short, int, double, float>, float>::value);
    STATIC_REQUIRE(!aV2P::meta::Contains<std::tuple<short, int, double, float>, char>::value);
}

TEST_CASE("isList", "[meta]")
{
    STATIC_REQUIRE(aV2P::meta::isList<std::tuple<int>>);
    STATIC_REQUIRE(aV2P::meta::isList<std::tuple<int, float>>);
    STATIC_REQUIRE_FALSE(aV2P::meta::isList<int>);

    STATIC_REQUIRE(aV2P::meta::isList<TypeList<int>>);
    STATIC_REQUIRE(aV2P::meta::isList<TypeList<int, float, double>>);
}

TEST_CASE("ToList", "[meta]")
{
    STATIC_REQUIRE(std::is_same_v<typename aV2P::meta::ToList<TypeList, int>::type, TypeList<int>>);
    STATIC_REQUIRE(
        std::is_same_v<typename aV2P::meta::ToList<TypeList, float, double, int>::type, TypeList<float, double, int>>);
    STATIC_REQUIRE(
        std::is_same_v<typename aV2P::meta::ToList<TypeList, TypeList<unsigned int>>::type, TypeList<unsigned int>>);
    STATIC_REQUIRE(std::is_same_v<
                   typename aV2P::meta::ToList<TypeList, TypeList<float, double, int>>::type,
                   TypeList<float, double, int>>);

    STATIC_REQUIRE(std::is_same_v<typename aV2P::meta::ToList<std::tuple, int>::type, std::tuple<int>>);
    STATIC_REQUIRE(
        std::is_same_v<typename aV2P::meta::ToList<std::tuple, std::tuple<float>>::type, std::tuple<float>>);
}

TEST_CASE("toTuple", "[meta]")
{
    STATIC_REQUIRE(std::is_same_v<aV2P::meta::ToTuple<int>, std::tuple<int>>);
    STATIC_REQUIRE(std::is_same_v<aV2P::meta::ToTuple<int, float, double>, std::tuple<int, float, double>>);
    STATIC_REQUIRE(std::is_same_v<aV2P::meta::ToTuple<std::tuple<int>>, std::tuple<int>>);
    STATIC_REQUIRE(
        std::is_same_v<aV2P::meta::ToTuple<std::tuple<int, float, double>>, std::tuple<int, float, double>>);
}

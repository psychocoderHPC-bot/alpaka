/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

// Regression test for the alpaka 2.2 public headers that aV2P has to keep available:
//   aV2P/atomic/AtomicAtomicRef.hpp
//   aV2P/intrinsic/IntrinsicFallback.hpp
// It exercises only the symbols those two 2.2 headers expose.

#include <aV2P/atomic/AtomicAtomicRef.hpp>
#include <aV2P/intrinsic/IntrinsicFallback.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>

TEST_CASE("atomicAtomicRef", "[atomic]")
{
    aV2P::AtomicAtomicRef atomic;
    aV2P::isSupportedByAtomicAtomicRef<int>();

    int value = 3;
    CHECK(aV2P::atomicAdd(atomic, &value, 4) == 3);
    CHECK(value == 7);
    CHECK(aV2P::atomicSub(atomic, &value, 2) == 7);
    CHECK(value == 5);
    CHECK(aV2P::atomicMin(atomic, &value, 2) == 5);
    CHECK(value == 2);
    CHECK(aV2P::atomicMax(atomic, &value, 9) == 2);
    CHECK(value == 9);
    CHECK(aV2P::atomicExch(atomic, &value, 1) == 9);
    CHECK(value == 1);

    value = 1;
    CHECK(aV2P::atomicInc(atomic, &value, 3) == 1);
    CHECK(value == 2);
    value = 3;
    CHECK(aV2P::atomicInc(atomic, &value, 3) == 3);
    CHECK(value == 0);
    value = 2;
    CHECK(aV2P::atomicDec(atomic, &value, 4) == 2);
    CHECK(value == 1);
    value = 0;
    CHECK(aV2P::atomicDec(atomic, &value, 4) == 0);
    CHECK(value == 4);

    value = 0b1100;
    CHECK(aV2P::atomicAnd(atomic, &value, 0b1010) == 0b1100);
    CHECK(value == 0b1000);
    CHECK(aV2P::atomicOr(atomic, &value, 0b0011) == 0b1000);
    CHECK(value == 0b1011);
    CHECK(aV2P::atomicXor(atomic, &value, 0b1111) == 0b1011);
    CHECK(value == 0b0100);

    value = 4;
    CHECK(aV2P::atomicCas(atomic, &value, 4, 6) == 4);
    CHECK(value == 6);
    CHECK(aV2P::atomicCas(atomic, &value, 4, 8) == 6);
    CHECK(value == 6);
}

TEST_CASE("intrinsicFallback", "[intrinsic]")
{
    aV2P::IntrinsicFallback intrinsic;

    CHECK(aV2P::popcount(intrinsic, std::uint32_t{0xF0F0'F0F0u}) == 16);
    CHECK(aV2P::popcount(intrinsic, std::uint64_t{0xF0F0'F0F0'F0F0'F0F0ull}) == 32);
    CHECK(aV2P::ffs(intrinsic, std::int32_t{0}) == 0);
    CHECK(aV2P::ffs(intrinsic, std::int32_t{8}) == 4);
    CHECK(aV2P::ffs(intrinsic, std::int64_t{1} << 40) == 41);
    CHECK(aV2P::ffs(intrinsic, std::int32_t{-1}) == 1);
}

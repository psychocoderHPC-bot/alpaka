/* Copyright 2023 Antonio Di Pilato, Jan Stephan, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/acc/Traits.hpp>
#include <aV2P/test/acc/TestAccs.hpp>
#include <aV2P/traits/Traits.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

TEMPLATE_LIST_TEST_CASE("NativeHandle", "[handle]", aV2P::test::TestAccs)
{
    using Dev = aV2P::Dev<TestType>;

    auto const platformAcc = aV2P::Platform<TestType>{};
    auto const devAcc = aV2P::getDevByIdx(platformAcc, 0);
    auto handle = aV2P::getNativeHandle(devAcc);

    STATIC_REQUIRE(std::is_same_v<aV2P::NativeHandle<Dev>, decltype(handle)>);
    // The const-reference form used by getNativeHandle() must yield the same type.
    STATIC_REQUIRE(std::is_same_v<aV2P::NativeHandle<Dev const&>, aV2P::NativeHandle<Dev>>);
    // The SYCL backend does not use an int as the native handle type
#ifndef AV2P_ACC_SYCL_ENABLED
    STATIC_REQUIRE(std::is_same_v<aV2P::NativeHandle<Dev>, int>);
    // Value identity for the first device: 2.2 returned the constant 0 for the CPU and the
    // device index (0) for CUDA/HIP. Both must survive the alpaka 3 uint32_t boundary.
    CHECK(handle == 0);
#endif
}

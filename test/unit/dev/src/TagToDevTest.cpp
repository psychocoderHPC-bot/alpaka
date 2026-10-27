/* Copyright 2025 Simone Balducci
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/dev/Traits.hpp>
#include <aV2P/test/acc/TestAccs.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

TEMPLATE_LIST_TEST_CASE("tagToDevice", "[dev]", aV2P::test::TestAccs)
{
    using Acc = TestType;
    using Tag = aV2P::AccToTag<Acc>;

    STATIC_REQUIRE(std::is_same_v<aV2P::Dev<Tag>, aV2P::Dev<Acc>>);
}

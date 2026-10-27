/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Bernhard Manfred Gruber, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/acc/Traits.hpp>
#include <aV2P/test/acc/TestAccs.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <iostream>

TEMPLATE_LIST_TEST_CASE("getAccName", "[acc]", aV2P::test::TestAccs)
{
    std::cout << aV2P::getAccName<TestType>() << std::endl;
}

/* Copyright 2022 Jiří Vyskočil, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/math/FloatEqualExact.hpp>
#include <aV2P/test/KernelExecutionFixture.hpp>
#include <aV2P/test/acc/TestAccs.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

struct FloatEqualExactTestKernel
{
    AV2P_NO_HOST_ACC_WARNING
    template<typename TAcc>
    AV2P_FN_ACC auto operator()(TAcc const& /* acc */, bool* success) const -> void
    {
        // Store the comparison result in a separate variable so that the function call is outside AV2P_CHECK.
        // In case AV2P_CHECK were ever somehow modified to silence the warning by itself.
        bool testValue = false;

        float floatValue = -1.0f;
        testValue = aV2P::math::floatEqualExactNoWarning(floatValue, -1.0f);
        AV2P_CHECK(*success, testValue);

        double doubleValue = -1.0;
        testValue = aV2P::math::floatEqualExactNoWarning(doubleValue, -1.0);
        AV2P_CHECK(*success, testValue);
    }
};

TEMPLATE_LIST_TEST_CASE("floatEqualExactTest", "[math]", aV2P::test::TestAccs)
{
    // Host tests

    // Store the comparison result in a separate variable so that the function call is outside REQUIRE.
    // In case REQUIRE were ever somehow modified to silence the warning by itself.
    bool testValue = false;

    float floatValue = -1.0f;
    testValue = aV2P::math::floatEqualExactNoWarning(floatValue, -1.0f);
    REQUIRE(testValue);

    double doubleValue = -1.0;
    testValue = aV2P::math::floatEqualExactNoWarning(doubleValue, -1.0);
    REQUIRE(testValue);

    // Device tests

    using Acc = TestType;
    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;

    aV2P::test::KernelExecutionFixture<Acc> fixture(aV2P::Vec<Dim, Idx>::ones());

    FloatEqualExactTestKernel kernelFloat;
    REQUIRE(fixture(kernelFloat));
}

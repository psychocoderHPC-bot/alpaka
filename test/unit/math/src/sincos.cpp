/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera, Bernhard Manfred Gruber,
 *                Sergei Bastrakov, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include "Defines.hpp"

#include <aV2P/math/Traits.hpp>
#include <aV2P/test/KernelExecutionFixture.hpp>
#include <aV2P/test/acc/TestAccs.hpp>
#include <aV2P/test/queue/Queue.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <type_traits>

struct SinCosTestKernel
{
    AV2P_NO_HOST_ACC_WARNING
    template<typename TAcc, typename FP>
    AV2P_FN_ACC auto operator()(TAcc const& acc, bool* success, FP const arg) const -> void
    {
        // if arg is hardcoded then compiler can optimize it out
        // (PTX kernel (float) was just empty)
        FP check_sin = aV2P::math::sin(acc, arg);
        FP check_cos = aV2P::math::cos(acc, arg);
        auto result_sin = FP{0};
        auto result_cos = FP{0};
        aV2P::math::sincos(acc, arg, result_sin, result_cos);
        AV2P_CHECK(
            *success,
            mathtest::almost_equal(acc, result_sin, check_sin, 1)
                && mathtest::almost_equal(acc, result_cos, check_cos, 1));
    }
};

using TestAccs = aV2P::test::EnabledAccs<aV2P::DimInt<1u>, std::size_t>;

TEMPLATE_LIST_TEST_CASE("sincos", "[sincos]", TestAccs)
{
    using Acc = TestType;
    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;

    aV2P::test::KernelExecutionFixture<Acc> fixture(aV2P::Vec<Dim, Idx>::ones());

    SinCosTestKernel kernel;

    REQUIRE(fixture(kernel, 0.42f)); // float
    REQUIRE(fixture(kernel, 0.42)); // double
    REQUIRE(fixture(kernel, aV2P::Complex<float>{0.35f, -0.24f})); // complex float
    REQUIRE(fixture(kernel, aV2P::Complex<double>{0.35, -0.24})); // complex double
}

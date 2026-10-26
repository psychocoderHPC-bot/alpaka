/* Copyright 2022 Jakob Krude, Benjamin Worpitz, Bernhard Manfred Gruber, Sergei Bastrakov, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include "Buffer.hpp"
#include "DataGen.hpp"
#include "Defines.hpp"
#include "Functor.hpp"

#include <aV2P/math/Complex.hpp>
#include <aV2P/test/KernelExecutionFixture.hpp>
#include <aV2P/test/acc/TestAccs.hpp>
#include <aV2P/test/queue/Queue.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using TestAccs = aV2P::test::EnabledAccs<aV2P::DimInt<1u>, std::size_t>;

namespace custom
{
    enum Custom
    {
        Abs,
        Acos,
        Acosh,
        Arg,
        Asin,
        Asinh,
        Atan,
        Atanh,
        Atan2,
        Cbrt,
        Ceil,
        Conj,
        Copysign,
        Cos,
        Cosh,
        Erf,
        Exp,
        Floor,
        Fma,
        Fmod,
        Log,
        Log2,
        Log10,
        Max,
        Min,
        Pow,
        Remainder,
        Round,
        Lround,
        Llround,
        Rsqrt,
        Sin,
        Sinh,
        Sincos,
        Sqrt,
        Tan,
        Tanh,
        Trunc,

        Arg1 = 1024,
        Arg2 = 2048,
        Arg3 = 4096,
    };

    AV2P_FN_HOST_ACC auto abs(Custom c);

    AV2P_FN_HOST_ACC auto abs(Custom c)
    {
        return Custom::Abs | c;
    }

    AV2P_FN_HOST_ACC auto acos(Custom c);

    AV2P_FN_HOST_ACC auto acos(Custom c)
    {
        return Custom::Acos | c;
    }

    AV2P_FN_HOST_ACC auto acosh(Custom c);

    AV2P_FN_HOST_ACC auto acosh(Custom c)
    {
        return Custom::Acosh | c;
    }

    AV2P_FN_HOST_ACC auto arg(Custom c);

    AV2P_FN_HOST_ACC auto arg(Custom c)
    {
        return Custom::Arg | c;
    }

    AV2P_FN_HOST_ACC auto asin(Custom c);

    AV2P_FN_HOST_ACC auto asin(Custom c)
    {
        return Custom::Asin | c;
    }

    AV2P_FN_HOST_ACC auto asinh(Custom c);

    AV2P_FN_HOST_ACC auto asinh(Custom c)
    {
        return Custom::Asinh | c;
    }

    AV2P_FN_HOST_ACC auto atan(Custom c);

    AV2P_FN_HOST_ACC auto atan(Custom c)
    {
        return Custom::Atan | c;
    }

    AV2P_FN_HOST_ACC auto atanh(Custom c);

    AV2P_FN_HOST_ACC auto atanh(Custom c)
    {
        return Custom::Atanh | c;
    }

    AV2P_FN_HOST_ACC auto atan2(Custom a, Custom b);

    AV2P_FN_HOST_ACC auto atan2(Custom a, Custom b)
    {
        return Custom::Atan2 | a | b;
    }

    AV2P_FN_HOST_ACC auto cbrt(Custom c);

    AV2P_FN_HOST_ACC auto cbrt(Custom c)
    {
        return Custom::Cbrt | c;
    }

    AV2P_FN_HOST_ACC auto ceil(Custom c);

    AV2P_FN_HOST_ACC auto ceil(Custom c)
    {
        return Custom::Ceil | c;
    }

    AV2P_FN_HOST_ACC auto conj(Custom c);

    AV2P_FN_HOST_ACC auto conj(Custom c)
    {
        return Custom::Conj | c;
    }

    AV2P_FN_HOST_ACC auto copysign(Custom a, Custom b);

    AV2P_FN_HOST_ACC auto copysign(Custom a, Custom b)
    {
        return Custom::Copysign | a | b;
    }

    AV2P_FN_HOST_ACC auto cos(Custom c);

    AV2P_FN_HOST_ACC auto cos(Custom c)
    {
        return Custom::Cos | c;
    }

    AV2P_FN_HOST_ACC auto cosh(Custom c);

    AV2P_FN_HOST_ACC auto cosh(Custom c)
    {
        return Custom::Cosh | c;
    }

    AV2P_FN_HOST_ACC auto erf(Custom c);

    AV2P_FN_HOST_ACC auto erf(Custom c)
    {
        return Custom::Erf | c;
    }

    AV2P_FN_HOST_ACC auto exp(Custom c);

    AV2P_FN_HOST_ACC auto exp(Custom c)
    {
        return Custom::Exp | c;
    }

    AV2P_FN_HOST_ACC auto floor(Custom c);

    AV2P_FN_HOST_ACC auto floor(Custom c)
    {
        return Custom::Floor | c;
    }

    AV2P_FN_HOST_ACC auto fma(Custom a, Custom b, Custom c);

    AV2P_FN_HOST_ACC auto fma(Custom a, Custom b, Custom c)
    {
        return Custom::Fma | a | b | c;
    }

    AV2P_FN_HOST_ACC auto fmod(Custom a, Custom b);

    AV2P_FN_HOST_ACC auto fmod(Custom a, Custom b)
    {
        return Custom::Fmod | a | b;
    }

    AV2P_FN_HOST_ACC auto log(Custom c);

    AV2P_FN_HOST_ACC auto log(Custom c)
    {
        return Custom::Log | c;
    }

    AV2P_FN_HOST_ACC auto log2(Custom c);

    AV2P_FN_HOST_ACC auto log2(Custom c)
    {
        return Custom::Log2 | c;
    }

    AV2P_FN_HOST_ACC auto log10(Custom c);

    AV2P_FN_HOST_ACC auto log10(Custom c)
    {
        return Custom::Log10 | c;
    }

    AV2P_FN_HOST_ACC auto max(Custom a, Custom b);

    AV2P_FN_HOST_ACC auto max(Custom a, Custom b)
    {
        return Custom::Max | a | b;
    }

    AV2P_FN_HOST_ACC auto min(Custom a, Custom b);

    AV2P_FN_HOST_ACC auto min(Custom a, Custom b)
    {
        return Custom::Min | a | b;
    }

    AV2P_FN_HOST_ACC auto pow(Custom a, Custom b);

    AV2P_FN_HOST_ACC auto pow(Custom a, Custom b)
    {
        return Custom::Pow | a | b;
    }

    AV2P_FN_HOST_ACC auto remainder(Custom a, Custom b);

    AV2P_FN_HOST_ACC auto remainder(Custom a, Custom b)
    {
        return Custom::Remainder | a | b;
    }

    AV2P_FN_HOST_ACC auto round(Custom c);

    AV2P_FN_HOST_ACC auto round(Custom c)
    {
        return Custom::Round | c;
    }

    AV2P_FN_HOST_ACC auto lround(Custom c);

    AV2P_FN_HOST_ACC auto lround(Custom c)
    {
        return Custom::Lround | c;
    }

    AV2P_FN_HOST_ACC auto llround(Custom c);

    AV2P_FN_HOST_ACC auto llround(Custom c)
    {
        return Custom::Llround | c;
    }

    AV2P_FN_HOST_ACC auto rsqrt(Custom c);

    AV2P_FN_HOST_ACC auto rsqrt(Custom c)
    {
        return Custom::Rsqrt | c;
    }

    AV2P_FN_HOST_ACC auto sin(Custom c);

    AV2P_FN_HOST_ACC auto sin(Custom c)
    {
        return Custom::Sin | c;
    }

    AV2P_FN_HOST_ACC auto sinh(Custom c);

    AV2P_FN_HOST_ACC auto sinh(Custom c)
    {
        return Custom::Sinh | c;
    }

    AV2P_FN_HOST_ACC void sincos(Custom c, Custom& a, Custom& b);

    AV2P_FN_HOST_ACC void sincos(Custom c, Custom& a, Custom& b)
    {
        a = static_cast<Custom>(Custom::Sincos | c | Custom::Arg2);
        b = static_cast<Custom>(Custom::Sincos | c | Custom::Arg3);
    }

    AV2P_FN_HOST_ACC auto sqrt(Custom c);

    AV2P_FN_HOST_ACC auto sqrt(Custom c)
    {
        return Custom::Sqrt | c;
    }

    AV2P_FN_HOST_ACC auto tan(Custom c);

    AV2P_FN_HOST_ACC auto tan(Custom c)
    {
        return Custom::Tan | c;
    }

    AV2P_FN_HOST_ACC auto tanh(Custom c);

    AV2P_FN_HOST_ACC auto tanh(Custom c)
    {
        return Custom::Tanh | c;
    }

    AV2P_FN_HOST_ACC auto trunc(Custom c);

    AV2P_FN_HOST_ACC auto trunc(Custom c)
    {
        return Custom::Trunc | c;
    }
} // namespace custom

struct AdlKernel
{
    template<typename Acc>
    AV2P_FN_ACC void operator()(Acc const& acc, bool* success) const noexcept
    {
        using custom::Custom;

        AV2P_CHECK(*success, aV2P::math::abs(acc, Custom::Arg1) == (Custom::Abs | Custom::Arg1));
        AV2P_CHECK(*success, aV2P::math::acos(acc, Custom::Arg1) == (Custom::Acos | Custom::Arg1));
        AV2P_CHECK(*success, aV2P::math::acosh(acc, Custom::Arg1) == (Custom::Acosh | Custom::Arg1));
        AV2P_CHECK(*success, aV2P::math::arg(acc, Custom::Arg1) == (Custom::Arg | Custom::Arg1));
        AV2P_CHECK(*success, aV2P::math::asin(acc, Custom::Arg1) == (Custom::Asin | Custom::Arg1));
        AV2P_CHECK(*success, aV2P::math::asinh(acc, Custom::Arg1) == (Custom::Asinh | Custom::Arg1));
        AV2P_CHECK(*success, aV2P::math::atan(acc, Custom::Arg1) == (Custom::Atan | Custom::Arg1));
        AV2P_CHECK(*success, aV2P::math::atanh(acc, Custom::Arg1) == (Custom::Atanh | Custom::Arg1));
        AV2P_CHECK(*success, aV2P::math::cbrt(acc, Custom::Arg1) == (Custom::Cbrt | Custom::Arg1));
        AV2P_CHECK(*success, aV2P::math::ceil(acc, Custom::Arg1) == (Custom::Ceil | Custom::Arg1));
        AV2P_CHECK(*success, aV2P::math::conj(acc, Custom::Arg1) == (Custom::Conj | Custom::Arg1));
        AV2P_CHECK(*success, aV2P::math::cos(acc, Custom::Arg1) == (Custom::Cos | Custom::Arg1));
        AV2P_CHECK(*success, aV2P::math::cosh(acc, Custom::Arg1) == (Custom::Cosh | Custom::Arg1));
        AV2P_CHECK(*success, aV2P::math::erf(acc, Custom::Arg1) == (Custom::Erf | Custom::Arg1));
        AV2P_CHECK(*success, aV2P::math::exp(acc, Custom::Arg1) == (Custom::Exp | Custom::Arg1));
        AV2P_CHECK(*success, aV2P::math::floor(acc, Custom::Arg1) == (Custom::Floor | Custom::Arg1));
        AV2P_CHECK(*success, aV2P::math::log(acc, Custom::Arg1) == (Custom::Log | Custom::Arg1));
        AV2P_CHECK(*success, aV2P::math::log2(acc, Custom::Arg1) == (Custom::Log2 | Custom::Arg1));
        AV2P_CHECK(*success, aV2P::math::log10(acc, Custom::Arg1) == (Custom::Log10 | Custom::Arg1));
        AV2P_CHECK(*success, aV2P::math::round(acc, Custom::Arg1) == (Custom::Round | Custom::Arg1));
        AV2P_CHECK(*success, aV2P::math::lround(acc, Custom::Arg1) == (Custom::Lround | Custom::Arg1));
        AV2P_CHECK(*success, aV2P::math::llround(acc, Custom::Arg1) == (Custom::Llround | Custom::Arg1));
        AV2P_CHECK(*success, aV2P::math::rsqrt(acc, Custom::Arg1) == (Custom::Rsqrt | Custom::Arg1));
        AV2P_CHECK(*success, aV2P::math::sin(acc, Custom::Arg1) == (Custom::Sin | Custom::Arg1));
        AV2P_CHECK(*success, aV2P::math::sinh(acc, Custom::Arg1) == (Custom::Sinh | Custom::Arg1));
        AV2P_CHECK(*success, aV2P::math::sqrt(acc, Custom::Arg1) == (Custom::Sqrt | Custom::Arg1));
        AV2P_CHECK(*success, aV2P::math::tan(acc, Custom::Arg1) == (Custom::Tan | Custom::Arg1));
        AV2P_CHECK(*success, aV2P::math::tanh(acc, Custom::Arg1) == (Custom::Tanh | Custom::Arg1));
        AV2P_CHECK(*success, aV2P::math::trunc(acc, Custom::Arg1) == (Custom::Trunc | Custom::Arg1));

        AV2P_CHECK(
            *success,
            aV2P::math::atan2(acc, Custom::Arg1, Custom::Arg2) == (Custom::Atan2 | Custom::Arg1 | Custom::Arg2));
        AV2P_CHECK(
            *success,
            aV2P::math::copysign(acc, Custom::Arg1, Custom::Arg2)
                == (Custom::Copysign | Custom::Arg1 | Custom::Arg2));
        AV2P_CHECK(
            *success,
            aV2P::math::fmod(acc, Custom::Arg1, Custom::Arg2) == (Custom::Fmod | Custom::Arg1 | Custom::Arg2));
        AV2P_CHECK(
            *success,
            aV2P::math::max(acc, Custom::Arg1, Custom::Arg2) == (Custom::Max | Custom::Arg1 | Custom::Arg2));
        AV2P_CHECK(
            *success,
            aV2P::math::min(acc, Custom::Arg1, Custom::Arg2) == (Custom::Min | Custom::Arg1 | Custom::Arg2));
        AV2P_CHECK(
            *success,
            aV2P::math::pow(acc, Custom::Arg1, Custom::Arg2) == (Custom::Pow | Custom::Arg1 | Custom::Arg2));
        AV2P_CHECK(
            *success,
            aV2P::math::remainder(acc, Custom::Arg1, Custom::Arg2)
                == (Custom::Remainder | Custom::Arg1 | Custom::Arg2));

        AV2P_CHECK(
            *success,
            aV2P::math::fma(acc, Custom::Arg1, Custom::Arg2, Custom::Arg3)
                == (Custom::Fma | Custom::Arg1 | Custom::Arg2 | Custom::Arg3));

        Custom a, b;
        aV2P::math::sincos(acc, Custom::Arg1, a, b);
        AV2P_CHECK(*success, a == (Custom::Sincos | Custom::Arg1 | Custom::Arg2));
        AV2P_CHECK(*success, b == (Custom::Sincos | Custom::Arg1 | Custom::Arg3));
    }
};

TEMPLATE_LIST_TEST_CASE("mathOps", "[math] [operator] [adl]", TestAccs)
{
    using Acc = TestType;
    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;
    auto fixture = aV2P::test::KernelExecutionFixture<Acc>{aV2P::Vec<Dim, Idx>::ones()};
    REQUIRE(fixture(AdlKernel{}));
}

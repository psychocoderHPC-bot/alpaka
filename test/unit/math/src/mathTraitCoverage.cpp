/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/math/MathUniformCudaHipBuiltIn.hpp>
#include <aV2P/math/Traits.hpp>
#include <aV2P/test/Check.hpp>
#include <aV2P/test/KernelExecutionFixture.hpp>
#include <aV2P/test/acc/TestAccs.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <complex>
#include <type_traits>

namespace math_coverage
{
    //! Minimal client vector type without any alpaka math support of its own.
    template<typename T>
    struct Vec
    {
        T value;
    };

    //! Returned by the client specializations below.
    struct Rejected
    {
    };
} // namespace math_coverage

/* Client specializations that make a claimed generic specialization visible.
 *
 * The conditions mirror what a client such as pmacc may provide: an unconditional `void` SFINAE
 * argument.  If the accelerator facade still claimed the argument type, both partial
 * specializations would match and `aV2P::math::<fn>(acc, arg)` would be ambiguous.  With the 2.2
 * conditions the client specialization is the only match.  The `requires` probes in the test cases
 * evaluate this without failing compilation.
 */
namespace aV2P::math::trait
{
    template<typename TCtx>
    struct Exp<TCtx, int, void>
    {
        math_coverage::Rejected operator()(TCtx const&, int) const;
    };

    template<typename TCtx>
    struct Abs<TCtx, unsigned int, void>
    {
        math_coverage::Rejected operator()(TCtx const&, unsigned int) const;
    };

    template<typename TCtx>
    struct Pow<TCtx, int, int, void>
    {
        math_coverage::Rejected operator()(TCtx const&, int, int) const;
    };

    template<typename TCtx>
    struct Fma<TCtx, int, int, int, void>
    {
        math_coverage::Rejected operator()(TCtx const&, int, int, int) const;
    };

    //! Client scalar reductions for a custom vector type (the pmacc pattern).
    template<typename TCtx, typename T>
    struct Exp<TCtx, math_coverage::Vec<T>, void>
    {
        AV2P_FN_HOST_ACC auto operator()(TCtx const& ctx, math_coverage::Vec<T> const& arg) const
            -> math_coverage::Vec<T>
        {
            return {aV2P::math::exp(ctx, arg.value)};
        }
    };

    template<typename TCtx, typename T>
    struct Pow<TCtx, math_coverage::Vec<T>, math_coverage::Vec<T>, void>
    {
        AV2P_FN_HOST_ACC auto operator()(
            TCtx const& ctx,
            math_coverage::Vec<T> const& base,
            math_coverage::Vec<T> const& exp) const -> math_coverage::Vec<T>
        {
            return {aV2P::math::pow(ctx, base.value, exp.value)};
        }
    };

    template<typename TCtx, typename T>
    struct Min<TCtx, math_coverage::Vec<T>, math_coverage::Vec<T>, void>
    {
        AV2P_FN_HOST_ACC auto operator()(
            TCtx const& ctx,
            math_coverage::Vec<T> const& lhs,
            math_coverage::Vec<T> const& rhs) const -> math_coverage::Vec<T>
        {
            return {aV2P::math::min(ctx, lhs.value, rhs.value)};
        }
    };

    template<typename TCtx, typename T>
    struct Max<TCtx, math_coverage::Vec<T>, math_coverage::Vec<T>, void>
    {
        AV2P_FN_HOST_ACC auto operator()(
            TCtx const& ctx,
            math_coverage::Vec<T> const& lhs,
            math_coverage::Vec<T> const& rhs) const -> math_coverage::Vec<T>
        {
            return {aV2P::math::max(ctx, lhs.value, rhs.value)};
        }
    };
} // namespace aV2P::math::trait

namespace
{
    /* Detection of the facade condition.
     *
     * The probes are positive when the client specialization above wins.  They are negative when the
     * generic accelerator specialization also matches, because the call would then be ambiguous.
     */
    template<typename TCtx>
    concept ExpRejectsInt = requires(TCtx const& ctx) { aV2P::math::exp(ctx, 1); };

    template<typename TCtx>
    concept AbsRejectsUnsigned = requires(TCtx const& ctx) { aV2P::math::abs(ctx, 1u); };

    template<typename TCtx>
    concept PowRejectsInt = requires(TCtx const& ctx) { aV2P::math::pow(ctx, 1, 1); };

    template<typename TCtx>
    concept FmaRejectsInt = requires(TCtx const& ctx) { aV2P::math::fma(ctx, 1, 1, 1); };

    struct VecMathKernel
    {
        template<typename TAcc>
        AV2P_FN_ACC auto operator()(TAcc const& acc, bool* success, float x) const -> void
        {
            using math_coverage::Vec;
            auto const e = aV2P::math::exp(acc, Vec<float>{x});
            AV2P_CHECK(*success, e.value > 0.0f);

            auto const p = aV2P::math::pow(acc, Vec<float>{x}, Vec<float>{2.0f});
            AV2P_CHECK(*success, p.value > 0.0f);

            auto const lo = aV2P::math::min(acc, Vec<float>{x}, Vec<float>{1.0f});
            AV2P_CHECK(*success, lo.value <= x);

            auto const hi = aV2P::math::max(acc, Vec<float>{x}, Vec<float>{1.0f});
            AV2P_CHECK(*success, hi.value >= x);
        }
    };
} // namespace

using namespace aV2P::math;
using TestAccs = aV2P::test::EnabledAccs<aV2P::DimInt<1u>, std::size_t>;

// clang-format off
#define CHECK_MARKER_TYPE(EXPR, TYPE)                                                                                 \
    STATIC_REQUIRE(std::is_same_v<std::decay_t<decltype(EXPR)>, TYPE>)

TEST_CASE("mathMarkerTraitsUnary", "[math][mathTraitCoverage]")
{
    CHECK_MARKER_TYPE(aV2P::math::abs(AbsUniformCudaHipBuiltIn{}, 1.5f), float);
    CHECK_MARKER_TYPE(aV2P::math::abs(AbsUniformCudaHipBuiltIn{}, 1.5), double);
    CHECK_MARKER_TYPE(aV2P::math::abs(AbsUniformCudaHipBuiltIn{}, 1), int);
    CHECK_MARKER_TYPE(aV2P::math::abs(AbsUniformCudaHipBuiltIn{}, 1L), long);

    CHECK_MARKER_TYPE(aV2P::math::acos(AcosUniformCudaHipBuiltIn{}, 1.5f), float);
    CHECK_MARKER_TYPE(aV2P::math::acos(AcosUniformCudaHipBuiltIn{}, 1.5), double);
    CHECK_MARKER_TYPE(aV2P::math::acosh(AcoshUniformCudaHipBuiltIn{}, 1.5f), float);
    CHECK_MARKER_TYPE(aV2P::math::acosh(AcoshUniformCudaHipBuiltIn{}, 1.5), double);
    CHECK_MARKER_TYPE(aV2P::math::arg(ArgUniformCudaHipBuiltIn{}, 1.5f), float);
    CHECK_MARKER_TYPE(aV2P::math::arg(ArgUniformCudaHipBuiltIn{}, 1.5), double);
    CHECK_MARKER_TYPE(aV2P::math::asin(AsinUniformCudaHipBuiltIn{}, 1.5f), float);
    CHECK_MARKER_TYPE(aV2P::math::asin(AsinUniformCudaHipBuiltIn{}, 1.5), double);
    CHECK_MARKER_TYPE(aV2P::math::asinh(AsinhUniformCudaHipBuiltIn{}, 1.5f), float);
    CHECK_MARKER_TYPE(aV2P::math::asinh(AsinhUniformCudaHipBuiltIn{}, 1.5), double);
    CHECK_MARKER_TYPE(aV2P::math::atan(AtanUniformCudaHipBuiltIn{}, 1.5f), float);
    CHECK_MARKER_TYPE(aV2P::math::atan(AtanUniformCudaHipBuiltIn{}, 1.5), double);
    CHECK_MARKER_TYPE(aV2P::math::atanh(AtanhUniformCudaHipBuiltIn{}, 1.5f), float);
    CHECK_MARKER_TYPE(aV2P::math::atanh(AtanhUniformCudaHipBuiltIn{}, 1.5), double);

    CHECK_MARKER_TYPE(aV2P::math::cbrt(CbrtUniformCudaHipBuiltIn{}, 1.5f), float);
    CHECK_MARKER_TYPE(aV2P::math::cbrt(CbrtUniformCudaHipBuiltIn{}, 1.5), double);
    CHECK_MARKER_TYPE(aV2P::math::cbrt(CbrtUniformCudaHipBuiltIn{}, 1), double);
    CHECK_MARKER_TYPE(aV2P::math::ceil(CeilUniformCudaHipBuiltIn{}, 1.5f), float);
    CHECK_MARKER_TYPE(aV2P::math::ceil(CeilUniformCudaHipBuiltIn{}, 1.5), double);
    {
        auto const cf = aV2P::math::conj(ConjUniformCudaHipBuiltIn{}, 1.5f);
        auto const cd = aV2P::math::conj(ConjUniformCudaHipBuiltIn{}, 1.5);
        STATIC_REQUIRE(std::is_same_v<std::decay_t<decltype(cf.real())>, float>);
        STATIC_REQUIRE(std::is_same_v<std::decay_t<decltype(cd.real())>, double>);
    }
    CHECK_MARKER_TYPE(aV2P::math::cos(CosUniformCudaHipBuiltIn{}, 1.5f), float);
    CHECK_MARKER_TYPE(aV2P::math::cos(CosUniformCudaHipBuiltIn{}, 1.5), double);
    CHECK_MARKER_TYPE(aV2P::math::cosh(CoshUniformCudaHipBuiltIn{}, 1.5f), float);
    CHECK_MARKER_TYPE(aV2P::math::cosh(CoshUniformCudaHipBuiltIn{}, 1.5), double);
    CHECK_MARKER_TYPE(aV2P::math::erf(ErfUniformCudaHipBuiltIn{}, 1.5f), float);
    CHECK_MARKER_TYPE(aV2P::math::erf(ErfUniformCudaHipBuiltIn{}, 1.5), double);
    CHECK_MARKER_TYPE(aV2P::math::exp(ExpUniformCudaHipBuiltIn{}, 1.5f), float);
    CHECK_MARKER_TYPE(aV2P::math::exp(ExpUniformCudaHipBuiltIn{}, 1.5), double);
    CHECK_MARKER_TYPE(aV2P::math::floor(FloorUniformCudaHipBuiltIn{}, 1.5f), float);
    CHECK_MARKER_TYPE(aV2P::math::floor(FloorUniformCudaHipBuiltIn{}, 1.5), double);

    CHECK_MARKER_TYPE(aV2P::math::isfinite(IsfiniteUniformCudaHipBuiltIn{}, 1.5f), bool);
    CHECK_MARKER_TYPE(aV2P::math::isfinite(IsfiniteUniformCudaHipBuiltIn{}, 1.5), bool);
    CHECK_MARKER_TYPE(aV2P::math::isinf(IsinfUniformCudaHipBuiltIn{}, 1.5f), bool);
    CHECK_MARKER_TYPE(aV2P::math::isinf(IsinfUniformCudaHipBuiltIn{}, 1.5), bool);
    CHECK_MARKER_TYPE(aV2P::math::isnan(IsnanUniformCudaHipBuiltIn{}, 1.5f), bool);
    CHECK_MARKER_TYPE(aV2P::math::isnan(IsnanUniformCudaHipBuiltIn{}, 1.5), bool);

    CHECK_MARKER_TYPE(aV2P::math::log(LogUniformCudaHipBuiltIn{}, 1.5f), float);
    CHECK_MARKER_TYPE(aV2P::math::log(LogUniformCudaHipBuiltIn{}, 1.5), double);
    CHECK_MARKER_TYPE(aV2P::math::log2(Log2UniformCudaHipBuiltIn{}, 1.5f), float);
    CHECK_MARKER_TYPE(aV2P::math::log2(Log2UniformCudaHipBuiltIn{}, 1.5), double);
    CHECK_MARKER_TYPE(aV2P::math::log10(Log10UniformCudaHipBuiltIn{}, 1.5f), float);
    CHECK_MARKER_TYPE(aV2P::math::log10(Log10UniformCudaHipBuiltIn{}, 1.5), double);

    CHECK_MARKER_TYPE(aV2P::math::round(RoundUniformCudaHipBuiltIn{}, 1.5f), float);
    CHECK_MARKER_TYPE(aV2P::math::round(RoundUniformCudaHipBuiltIn{}, 1.5), double);
    CHECK_MARKER_TYPE(aV2P::math::lround(RoundUniformCudaHipBuiltIn{}, 1.5f), long);
    CHECK_MARKER_TYPE(aV2P::math::lround(RoundUniformCudaHipBuiltIn{}, 1.5), long);
    CHECK_MARKER_TYPE(aV2P::math::llround(RoundUniformCudaHipBuiltIn{}, 1.5f), long long);
    CHECK_MARKER_TYPE(aV2P::math::llround(RoundUniformCudaHipBuiltIn{}, 1.5), long long);

    CHECK_MARKER_TYPE(aV2P::math::rsqrt(RsqrtUniformCudaHipBuiltIn{}, 1.5f), float);
    CHECK_MARKER_TYPE(aV2P::math::rsqrt(RsqrtUniformCudaHipBuiltIn{}, 1.5), double);
    CHECK_MARKER_TYPE(aV2P::math::sin(SinUniformCudaHipBuiltIn{}, 1.5f), float);
    CHECK_MARKER_TYPE(aV2P::math::sin(SinUniformCudaHipBuiltIn{}, 1.5), double);
    CHECK_MARKER_TYPE(aV2P::math::sinh(SinhUniformCudaHipBuiltIn{}, 1.5f), float);
    CHECK_MARKER_TYPE(aV2P::math::sinh(SinhUniformCudaHipBuiltIn{}, 1.5), double);
    CHECK_MARKER_TYPE(aV2P::math::sqrt(SqrtUniformCudaHipBuiltIn{}, 1.5f), float);
    CHECK_MARKER_TYPE(aV2P::math::sqrt(SqrtUniformCudaHipBuiltIn{}, 1.5), double);
    CHECK_MARKER_TYPE(aV2P::math::sqrt(SqrtUniformCudaHipBuiltIn{}, 1), double);
    CHECK_MARKER_TYPE(aV2P::math::tan(TanUniformCudaHipBuiltIn{}, 1.5f), float);
    CHECK_MARKER_TYPE(aV2P::math::tan(TanUniformCudaHipBuiltIn{}, 1.5), double);
    CHECK_MARKER_TYPE(aV2P::math::tanh(TanhUniformCudaHipBuiltIn{}, 1.5f), float);
    CHECK_MARKER_TYPE(aV2P::math::tanh(TanhUniformCudaHipBuiltIn{}, 1.5), double);
    CHECK_MARKER_TYPE(aV2P::math::trunc(TruncUniformCudaHipBuiltIn{}, 1.5f), float);
    CHECK_MARKER_TYPE(aV2P::math::trunc(TruncUniformCudaHipBuiltIn{}, 1.5), double);
}

TEST_CASE("mathMarkerTraitsBinary", "[math][mathTraitCoverage]")
{
    CHECK_MARKER_TYPE(aV2P::math::atan2(Atan2UniformCudaHipBuiltIn{}, 1.5f, 1.5f), float);
    CHECK_MARKER_TYPE(aV2P::math::atan2(Atan2UniformCudaHipBuiltIn{}, 1.5, 1.5), double);
    CHECK_MARKER_TYPE(aV2P::math::atan2(Atan2UniformCudaHipBuiltIn{}, 1.5f, 1.5), double);
    CHECK_MARKER_TYPE(aV2P::math::atan2(Atan2UniformCudaHipBuiltIn{}, 1.5, 1.5f), double);

    CHECK_MARKER_TYPE(aV2P::math::copysign(CopysignUniformCudaHipBuiltIn{}, 1.5f, 1.5f), float);
    CHECK_MARKER_TYPE(aV2P::math::copysign(CopysignUniformCudaHipBuiltIn{}, 1.5f, 1.5), double);
    CHECK_MARKER_TYPE(aV2P::math::copysign(CopysignUniformCudaHipBuiltIn{}, 1.5, 1.5), double);

    CHECK_MARKER_TYPE(aV2P::math::fmod(FmodUniformCudaHipBuiltIn{}, 1.5f, 1.5f), float);
    CHECK_MARKER_TYPE(aV2P::math::fmod(FmodUniformCudaHipBuiltIn{}, 1.5f, 1.5), double);
    CHECK_MARKER_TYPE(aV2P::math::fmod(FmodUniformCudaHipBuiltIn{}, 1.5, 1.5), double);

    CHECK_MARKER_TYPE(aV2P::math::pow(PowUniformCudaHipBuiltIn{}, 1.5f, 1.5f), float);
    CHECK_MARKER_TYPE(aV2P::math::pow(PowUniformCudaHipBuiltIn{}, 1.5, 1.5), double);
    CHECK_MARKER_TYPE(aV2P::math::pow(PowUniformCudaHipBuiltIn{}, 1.5f, 1.5), double);
    CHECK_MARKER_TYPE(aV2P::math::pow(PowUniformCudaHipBuiltIn{}, 1.5, 1.5f), double);

    CHECK_MARKER_TYPE(aV2P::math::remainder(RemainderUniformCudaHipBuiltIn{}, 1.5f, 1.5f), float);
    CHECK_MARKER_TYPE(aV2P::math::remainder(RemainderUniformCudaHipBuiltIn{}, 1.5f, 1.5), double);
    CHECK_MARKER_TYPE(aV2P::math::remainder(RemainderUniformCudaHipBuiltIn{}, 1.5, 1.5), double);

    CHECK_MARKER_TYPE(aV2P::math::fma(FmaUniformCudaHipBuiltIn{}, 1.5f, 1.5f, 1.5f), float);
    CHECK_MARKER_TYPE(aV2P::math::fma(FmaUniformCudaHipBuiltIn{}, 1.5, 1.5, 1.5), double);
    CHECK_MARKER_TYPE(aV2P::math::fma(FmaUniformCudaHipBuiltIn{}, 1.5f, 1.5, 1.5f), double);
}

TEST_CASE("mathMarkerTraitsSinCos", "[math][mathTraitCoverage]")
{
    float sinF = 0.0f;
    float cosF = 0.0f;
    double sinD = 0.0;
    double cosD = 0.0;
    STATIC_REQUIRE(std::is_same_v<
                   decltype(aV2P::math::sincos(SinCosUniformCudaHipBuiltIn{}, 1.5f, sinF, cosF)),
                   void>);
    STATIC_REQUIRE(std::is_same_v<
                   decltype(aV2P::math::sincos(SinCosUniformCudaHipBuiltIn{}, 1.5, sinD, cosD)),
                   void>);
    aV2P::math::sincos(SinCosUniformCudaHipBuiltIn{}, 1.0f, sinF, cosF);
    aV2P::math::sincos(SinCosUniformCudaHipBuiltIn{}, 1.0, sinD, cosD);
    REQUIRE(sinF > 0.0f);
    REQUIRE(cosF > 0.0f);
    REQUIRE(sinD > 0.0);
    REQUIRE(cosD > 0.0);
}

TEMPLATE_LIST_TEST_CASE("mathFacadeRejectsNonScalarArguments", "[math][mathTraitCoverage]", TestAccs)
{
    using Acc = TestType;
    STATIC_REQUIRE(ExpRejectsInt<Acc>);
    STATIC_REQUIRE(AbsRejectsUnsigned<Acc>);
    STATIC_REQUIRE(PowRejectsInt<Acc>);
    STATIC_REQUIRE(FmaRejectsInt<Acc>);
}

TEMPLATE_LIST_TEST_CASE("mathClientVectorSpecializationsAreUnambiguous", "[math][mathTraitCoverage]", TestAccs)
{
    using Acc = TestType;

    aV2P::test::KernelExecutionFixture<Acc> fixture(aV2P::Vec<aV2P::Dim<Acc>, aV2P::Idx<Acc>>::ones());
    REQUIRE(fixture(VecMathKernel{}, 2.0f));
}

// clang-format on

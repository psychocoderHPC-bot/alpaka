/* Copyright 2022 Jakob Krude, Benjamin Worpitz, Jan Stephan, Sergei Bastrakov
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "Defines.hpp"

#include <aV2P/alpaka.hpp>

#include <complex>
#include <type_traits>

namespace mathtest
{
    //! Helper trait to define a type conversion before passing T to a std:: math function
    //!
    //! The default implementation does no conversion
    //!
    //! @tparam T input type
    template<typename T>
    struct StdLibType
    {
        using type = T;
    };

    //! Specialization converting aV2P::Complex<T> to std::complex<T>
    template<typename T>
    struct StdLibType<aV2P::Complex<T>>
    {
        using type = std::complex<T>;
    };

// Can be used with operator() that will use either the std. function or the
// equivalent alpaka function (if an accelerator is passed additionally).
//! @param NAME The Name used for the Functor, e.g. OpAbs
//! @param ARITY Enum-type can be one ... n
//! @param STD_OP Function used for the host side, e.g. std::abs
//! @param AV2P_OP Function used for the device side, e.g. aV2P::math::abs.
//! @param ... List of Ranges. Needs to match the arity.
#define AV2P_TEST_MATH_OP_FUNCTOR(NAME, ARITY, STD_OP, AV2P_OP, ...)                                                  \
    struct NAME                                                                                                       \
    {                                                                                                                 \
        /* ranges is not a constexpr, so that it's accessible via for loop*/                                          \
        static constexpr Arity arity = ARITY;                                                                         \
        static constexpr size_t arity_nr = static_cast<size_t>(ARITY);                                                \
        Range ranges[arity_nr] = {__VA_ARGS__};                                                                       \
                                                                                                                      \
        AV2P_NO_HOST_ACC_WARNING                                                                                      \
        template<                                                                                                     \
            typename TAcc,                                                                                            \
            typename... TArgs, /* SFINAE: Enables if called from device. */                                           \
            typename std::enable_if_t<!std::is_same_v<TAcc, std::nullptr_t>, int> = 0>                                \
        AV2P_FN_ACC auto execute(TAcc const& acc, [[maybe_unused]] TArgs const&... args) const                        \
        {                                                                                                             \
            return AV2P_OP(acc, args...);                                                                             \
        }                                                                                                             \
                                                                                                                      \
        AV2P_NO_HOST_ACC_WARNING                                                                                      \
        template<                                                                                                     \
            typename TAcc = std::nullptr_t,                                                                           \
            typename TArg1, /* SFINAE: Enables if called from host. */                                                \
            typename std::enable_if_t<std::is_same_v<TAcc, std::nullptr_t>, int> = 0>                                 \
        AV2P_FN_HOST auto execute([[maybe_unused]] TAcc const& acc, TArg1 const& arg1) const                          \
        {                                                                                                             \
            return STD_OP(typename StdLibType<TArg1>::type(arg1));                                                    \
        }                                                                                                             \
                                                                                                                      \
        AV2P_NO_HOST_ACC_WARNING                                                                                      \
        template<                                                                                                     \
            typename TAcc = std::nullptr_t,                                                                           \
            typename TArg1,                                                                                           \
            typename TArg2, /* SFINAE: Enables if called from host. */                                                \
            typename std::enable_if_t<std::is_same_v<TAcc, std::nullptr_t>, int> = 0>                                 \
        AV2P_FN_HOST auto execute([[maybe_unused]] TAcc const& acc, TArg1 const& arg1, TArg2 const& arg2) const       \
        {                                                                                                             \
            return STD_OP(typename StdLibType<TArg1>::type(arg1), typename StdLibType<TArg2>::type(arg2));            \
        }                                                                                                             \
                                                                                                                      \
        AV2P_NO_HOST_ACC_WARNING                                                                                      \
        template<                                                                                                     \
            typename TAcc = std::nullptr_t,                                                                           \
            typename TArg1,                                                                                           \
            typename TArg2,                                                                                           \
            typename TArg3, /* SFINAE: Enables if called from host. */                                                \
            typename std::enable_if_t<std::is_same_v<TAcc, std::nullptr_t>, int> = 0>                                 \
        AV2P_FN_HOST auto execute(                                                                                    \
            [[maybe_unused]] TAcc const& acc,                                                                         \
            TArg1 const& arg1,                                                                                        \
            TArg2 const& arg2,                                                                                        \
            TArg3 const& arg3) const                                                                                  \
        {                                                                                                             \
            return STD_OP(                                                                                            \
                typename StdLibType<TArg1>::type(arg1),                                                               \
                typename StdLibType<TArg2>::type(arg2),                                                               \
                typename StdLibType<TArg3>::type(arg3));                                                              \
        }                                                                                                             \
                                                                                                                      \
        /* assigns args by arity */                                                                                   \
        AV2P_NO_HOST_ACC_WARNING                                                                                      \
        template<typename T, typename TAcc = std::nullptr_t>                                                          \
        AV2P_FN_HOST_ACC auto operator()(ArgsItem<T, Arity::Unary> const& args, TAcc const& acc = nullptr) const      \
        {                                                                                                             \
            return execute(acc, args.arg[0]);                                                                         \
        }                                                                                                             \
                                                                                                                      \
        /* assigns args by arity */                                                                                   \
        AV2P_NO_HOST_ACC_WARNING                                                                                      \
        template<typename T, typename TAcc = std::nullptr_t>                                                          \
        AV2P_FN_HOST_ACC auto operator()(ArgsItem<T, Arity::Binary> const& args, TAcc const& acc = nullptr) const     \
        {                                                                                                             \
            return execute(acc, args.arg[0], args.arg[1]);                                                            \
        }                                                                                                             \
                                                                                                                      \
        /* assigns args by arity */                                                                                   \
        AV2P_NO_HOST_ACC_WARNING                                                                                      \
        template<typename T, typename TAcc = std::nullptr_t>                                                          \
        AV2P_FN_HOST_ACC auto operator()(ArgsItem<T, Arity::Ternary> const& args, TAcc const& acc = nullptr) const    \
        {                                                                                                             \
            return execute(acc, args.arg[0], args.arg[1], args.arg[2]);                                               \
        }                                                                                                             \
                                                                                                                      \
        friend std::ostream& operator<<(std::ostream& out, const NAME& /* op */)                                      \
        {                                                                                                             \
            out << #NAME;                                                                                             \
            return out;                                                                                               \
        }                                                                                                             \
    };


    AV2P_TEST_MATH_OP_FUNCTOR(OpAbs, Arity::Unary, std::abs, aV2P::math::abs, Range::Unrestricted)

    AV2P_TEST_MATH_OP_FUNCTOR(OpAcos, Arity::Unary, std::acos, aV2P::math::acos, Range::OneNeighbourhood)

    AV2P_TEST_MATH_OP_FUNCTOR(OpAcosh, Arity::Unary, std::acosh, aV2P::math::acosh, Range::PositiveOnly)

    AV2P_TEST_MATH_OP_FUNCTOR(OpArg, Arity::Unary, std::arg, aV2P::math::arg, Range::Unrestricted)

    AV2P_TEST_MATH_OP_FUNCTOR(OpAsin, Arity::Unary, std::asin, aV2P::math::asin, Range::OneNeighbourhood)

    AV2P_TEST_MATH_OP_FUNCTOR(OpAsinh, Arity::Unary, std::asinh, aV2P::math::asinh, Range::Unrestricted)

    AV2P_TEST_MATH_OP_FUNCTOR(OpAtan, Arity::Unary, std::atan, aV2P::math::atan, Range::Unrestricted)

    AV2P_TEST_MATH_OP_FUNCTOR(OpAtanh, Arity::Unary, std::atanh, aV2P::math::atanh, Range::OneNeighbourhood)

    AV2P_TEST_MATH_OP_FUNCTOR(OpCbrt, Arity::Unary, std::cbrt, aV2P::math::cbrt, Range::Unrestricted)

    AV2P_TEST_MATH_OP_FUNCTOR(OpCeil, Arity::Unary, std::ceil, aV2P::math::ceil, Range::Unrestricted)

    AV2P_TEST_MATH_OP_FUNCTOR(OpCos, Arity::Unary, std::cos, aV2P::math::cos, Range::Unrestricted)

    AV2P_TEST_MATH_OP_FUNCTOR(OpCosh, Arity::Unary, std::cosh, aV2P::math::cosh, Range::Unrestricted)

    AV2P_TEST_MATH_OP_FUNCTOR(OpErf, Arity::Unary, std::erf, aV2P::math::erf, Range::Unrestricted)

    AV2P_TEST_MATH_OP_FUNCTOR(OpExp, Arity::Unary, std::exp, aV2P::math::exp, Range::Unrestricted)

    AV2P_TEST_MATH_OP_FUNCTOR(OpFloor, Arity::Unary, std::floor, aV2P::math::floor, Range::Unrestricted)

    AV2P_TEST_MATH_OP_FUNCTOR(OpLog, Arity::Unary, std::log, aV2P::math::log, Range::PositiveOnly)

    AV2P_TEST_MATH_OP_FUNCTOR(OpLog2, Arity::Unary, std::log2, aV2P::math::log2, Range::PositiveOnly)

    AV2P_TEST_MATH_OP_FUNCTOR(OpLog10, Arity::Unary, std::log10, aV2P::math::log10, Range::PositiveOnly)

    AV2P_TEST_MATH_OP_FUNCTOR(OpRound, Arity::Unary, std::round, aV2P::math::round, Range::Unrestricted)

    // There is no std implementation, look in Defines.hpp.
    AV2P_TEST_MATH_OP_FUNCTOR(OpRsqrt, Arity::Unary, rsqrt, aV2P::math::rsqrt, Range::PositiveOnly)

    AV2P_TEST_MATH_OP_FUNCTOR(OpSin, Arity::Unary, std::sin, aV2P::math::sin, Range::Unrestricted)

    AV2P_TEST_MATH_OP_FUNCTOR(OpSinh, Arity::Unary, std::sinh, aV2P::math::sinh, Range::Unrestricted)

    AV2P_TEST_MATH_OP_FUNCTOR(OpSqrt, Arity::Unary, std::sqrt, aV2P::math::sqrt, Range::PositiveAndZero)

    AV2P_TEST_MATH_OP_FUNCTOR(OpTan, Arity::Unary, std::tan, aV2P::math::tan, Range::Unrestricted)

    AV2P_TEST_MATH_OP_FUNCTOR(OpTanh, Arity::Unary, std::tanh, aV2P::math::tanh, Range::Unrestricted)

    AV2P_TEST_MATH_OP_FUNCTOR(OpTrunc, Arity::Unary, std::trunc, aV2P::math::trunc, Range::Unrestricted)

    AV2P_TEST_MATH_OP_FUNCTOR(OpIsnan, Arity::Unary, std::isnan, aV2P::math::isnan, Range::Anything)

    AV2P_TEST_MATH_OP_FUNCTOR(OpIsinf, Arity::Unary, std::isinf, aV2P::math::isinf, Range::Anything)

    AV2P_TEST_MATH_OP_FUNCTOR(OpIsfinite, Arity::Unary, std::isfinite, aV2P::math::isfinite, Range::Anything)

    // All binary operators.
    AV2P_TEST_MATH_OP_FUNCTOR(OpAtan2, Arity::Binary, std::atan2, aV2P::math::atan2, Range::NotZero, Range::NotZero)

    AV2P_TEST_MATH_OP_FUNCTOR(
        OpCopysign,
        Arity::Binary,
        std::copysign,
        aV2P::math::copysign,
        Range::Unrestricted,
        Range::Unrestricted)

    AV2P_TEST_MATH_OP_FUNCTOR(OpFmod, Arity::Binary, std::fmod, aV2P::math::fmod, Range::Unrestricted, Range::NotZero)

    AV2P_TEST_MATH_OP_FUNCTOR(
        OpMax,
        Arity::Binary,
        std::max,
        aV2P::math::max,
        Range::Unrestricted,
        Range::Unrestricted)

    AV2P_TEST_MATH_OP_FUNCTOR(
        OpMin,
        Arity::Binary,
        std::min,
        aV2P::math::min,
        Range::Unrestricted,
        Range::Unrestricted)

    AV2P_TEST_MATH_OP_FUNCTOR(
        OpPow,
        Arity::Binary,
        std::pow,
        aV2P::math::pow,
        Range::PositiveAndZero,
        Range::Unrestricted)

    AV2P_TEST_MATH_OP_FUNCTOR(
        OpRemainder,
        Arity::Binary,
        std::remainder,
        aV2P::math::remainder,
        Range::Unrestricted,
        Range::NotZero)

    // All ternary operators.
    AV2P_TEST_MATH_OP_FUNCTOR(
        OpFma,
        Arity::Ternary,
        std::fma,
        aV2P::math::fma,
        Range::Unrestricted,
        Range::Unrestricted,
        Range::Unrestricted)

    // Unary functors to be used only for real types
    using UnaryFunctorsReal = std::tuple<
        OpAbs,
        OpAcos,
        OpAcosh,
        OpArg,
        OpAsin,
        OpAsinh,
        OpAtan,
        OpAtanh,
        OpCbrt,
        OpCeil,
        OpCos,
        OpCosh,
        OpErf,
        OpExp,
        OpFloor,
        OpLog,
        OpLog2,
        OpLog10,
        OpRound,
        OpRsqrt,
        OpSin,
        OpSinh,
        OpSqrt,
        OpTan,
        OpTanh,
        OpTrunc,
        OpIsnan,
        OpIsinf,
        OpIsfinite>;

    // Binary functors to be used only for real types
    using BinaryFunctorsReal = std::tuple<OpAtan2, OpCopysign, OpFmod, OpMax, OpMin, OpPow, OpRemainder>;

    // Ternary functors to be used only for real types
    using TernaryFunctorsReal = std::tuple<OpFma>;

    // For complex numbers also test arithmetic operations
    AV2P_TEST_MATH_OP_FUNCTOR(OpDivides, Arity::Binary, std::divides<>{}, divides, Range::Unrestricted, Range::NotZero)

    AV2P_TEST_MATH_OP_FUNCTOR(OpMinus, Arity::Binary, std::minus<>{}, minus, Range::Unrestricted, Range::Unrestricted)

    AV2P_TEST_MATH_OP_FUNCTOR(
        OpMultiplies,
        Arity::Binary,
        std::multiplies<>{},
        multiplies,
        Range::Unrestricted,
        Range::Unrestricted)

    AV2P_TEST_MATH_OP_FUNCTOR(OpPlus, Arity::Binary, std::plus<>{}, plus, Range::Unrestricted, Range::Unrestricted)

    // conj() is only tested for complex as it returns a complex type for real arguments
    // and it doesn't fit the existing tests' infrastructure
    AV2P_TEST_MATH_OP_FUNCTOR(OpConj, Arity::Unary, std::conj, aV2P::math::conj, Range::Unrestricted)

    // As a workaround for complex 0^0 unit tests issues, test complex pow for positive range only
    AV2P_TEST_MATH_OP_FUNCTOR(
        OpPowComplex,
        Arity::Binary,
        std::pow,
        aV2P::math::pow,
        Range::PositiveOnly,
        Range::Unrestricted)

    // Unary functors to be used for both real and complex types
    using UnaryFunctorsComplex = std::tuple<
        OpAbs,
        OpAcos,
        OpAcosh,
        OpArg,
        OpAsin,
        OpAsinh,
        OpAtan,
        OpAtanh,
        OpConj,
        OpCos,
        OpCosh,
        OpExp,
        OpLog,
        OpLog10,
        OpRsqrt,
        OpSin,
        OpSinh,
        OpSqrt,
        OpTan,
        OpTanh>;

    // Binary functors to be used for complex types
    using BinaryFunctorsComplex = std::tuple<OpDivides, OpMinus, OpMultiplies, OpPlus, OpPowComplex>;
} // namespace mathtest

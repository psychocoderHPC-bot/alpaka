/* Copyright 2024 Axel Huebl, Benjamin Worpitz, Matthias Werner, Bert Wesarg, Valentin Gehrke, René Widera,
 * Jan Stephan, Andrea Bocci, Bernhard Manfred Gruber, Jeffrey Kelling, Sergei Bastrakov
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Common.hpp"
#include "aV2P/core/Interface.hpp"
#include "aV2P/math/Traits.hpp"

#include <alpaka/alpaka.hpp>

#include <type_traits>

namespace aV2P::math
{
    //! The CUDA built in abs.
    class AbsUniformCudaHipBuiltIn : public interface::Implements<ConceptMathAbs, AbsUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in acos.
    class AcosUniformCudaHipBuiltIn : public interface::Implements<ConceptMathAcos, AcosUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in acosh.
    class AcoshUniformCudaHipBuiltIn : public interface::Implements<ConceptMathAcosh, AcoshUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in arg.
    class ArgUniformCudaHipBuiltIn : public interface::Implements<ConceptMathArg, ArgUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in asin.
    class AsinUniformCudaHipBuiltIn : public interface::Implements<ConceptMathAsin, AsinUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in asinh.
    class AsinhUniformCudaHipBuiltIn : public interface::Implements<ConceptMathAsinh, AsinhUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in atan.
    class AtanUniformCudaHipBuiltIn : public interface::Implements<ConceptMathAtan, AtanUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in atanh.
    class AtanhUniformCudaHipBuiltIn : public interface::Implements<ConceptMathAtanh, AtanhUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in atan2.
    class Atan2UniformCudaHipBuiltIn : public interface::Implements<ConceptMathAtan2, Atan2UniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in cbrt.
    class CbrtUniformCudaHipBuiltIn : public interface::Implements<ConceptMathCbrt, CbrtUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in ceil.
    class CeilUniformCudaHipBuiltIn : public interface::Implements<ConceptMathCeil, CeilUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in conj.
    class ConjUniformCudaHipBuiltIn : public interface::Implements<ConceptMathConj, ConjUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in copysign.
    class CopysignUniformCudaHipBuiltIn
        : public interface::Implements<ConceptMathCopysign, CopysignUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in cos.
    class CosUniformCudaHipBuiltIn : public interface::Implements<ConceptMathCos, CosUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in cosh.
    class CoshUniformCudaHipBuiltIn : public interface::Implements<ConceptMathCosh, CoshUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in erf.
    class ErfUniformCudaHipBuiltIn : public interface::Implements<ConceptMathErf, ErfUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in exp.
    class ExpUniformCudaHipBuiltIn : public interface::Implements<ConceptMathExp, ExpUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in floor.
    class FloorUniformCudaHipBuiltIn : public interface::Implements<ConceptMathFloor, FloorUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in fma.
    class FmaUniformCudaHipBuiltIn : public interface::Implements<ConceptMathFma, FmaUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in fmod.
    class FmodUniformCudaHipBuiltIn : public interface::Implements<ConceptMathFmod, FmodUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in isfinite.
    class IsfiniteUniformCudaHipBuiltIn
        : public interface::Implements<ConceptMathIsfinite, IsfiniteUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in isinf.
    class IsinfUniformCudaHipBuiltIn : public interface::Implements<ConceptMathIsinf, IsinfUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in isnan.
    class IsnanUniformCudaHipBuiltIn : public interface::Implements<ConceptMathIsnan, IsnanUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in log.
    class LogUniformCudaHipBuiltIn : public interface::Implements<ConceptMathLog, LogUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in log2.
    class Log2UniformCudaHipBuiltIn : public interface::Implements<ConceptMathLog2, Log2UniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in log10.
    class Log10UniformCudaHipBuiltIn : public interface::Implements<ConceptMathLog10, Log10UniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in max.
    class MaxUniformCudaHipBuiltIn : public interface::Implements<ConceptMathMax, MaxUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in min.
    class MinUniformCudaHipBuiltIn : public interface::Implements<ConceptMathMin, MinUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in pow.
    class PowUniformCudaHipBuiltIn : public interface::Implements<ConceptMathPow, PowUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in remainder.
    class RemainderUniformCudaHipBuiltIn
        : public interface::Implements<ConceptMathRemainder, RemainderUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA round.
    class RoundUniformCudaHipBuiltIn : public interface::Implements<ConceptMathRound, RoundUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA rsqrt.
    class RsqrtUniformCudaHipBuiltIn : public interface::Implements<ConceptMathRsqrt, RsqrtUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA sin.
    class SinUniformCudaHipBuiltIn : public interface::Implements<ConceptMathSin, SinUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA sinh.
    class SinhUniformCudaHipBuiltIn : public interface::Implements<ConceptMathSinh, SinhUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA sincos.
    class SinCosUniformCudaHipBuiltIn : public interface::Implements<ConceptMathSinCos, SinCosUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA sqrt.
    class SqrtUniformCudaHipBuiltIn : public interface::Implements<ConceptMathSqrt, SqrtUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA tan.
    class TanUniformCudaHipBuiltIn : public interface::Implements<ConceptMathTan, TanUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA tanh.
    class TanhUniformCudaHipBuiltIn : public interface::Implements<ConceptMathTanh, TanhUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA trunc.
    class TruncUniformCudaHipBuiltIn : public interface::Implements<ConceptMathTrunc, TruncUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA/HIP built in math trait specializations.
    class MathUniformCudaHipBuiltIn
        : public AbsUniformCudaHipBuiltIn
        , public AcosUniformCudaHipBuiltIn
        , public AcoshUniformCudaHipBuiltIn
        , public ArgUniformCudaHipBuiltIn
        , public AsinUniformCudaHipBuiltIn
        , public AsinhUniformCudaHipBuiltIn
        , public AtanUniformCudaHipBuiltIn
        , public AtanhUniformCudaHipBuiltIn
        , public Atan2UniformCudaHipBuiltIn
        , public CbrtUniformCudaHipBuiltIn
        , public CeilUniformCudaHipBuiltIn
        , public ConjUniformCudaHipBuiltIn
        , public CopysignUniformCudaHipBuiltIn
        , public CosUniformCudaHipBuiltIn
        , public CoshUniformCudaHipBuiltIn
        , public ErfUniformCudaHipBuiltIn
        , public ExpUniformCudaHipBuiltIn
        , public FloorUniformCudaHipBuiltIn
        , public FmaUniformCudaHipBuiltIn
        , public FmodUniformCudaHipBuiltIn
        , public LogUniformCudaHipBuiltIn
        , public Log2UniformCudaHipBuiltIn
        , public Log10UniformCudaHipBuiltIn
        , public MaxUniformCudaHipBuiltIn
        , public MinUniformCudaHipBuiltIn
        , public PowUniformCudaHipBuiltIn
        , public RemainderUniformCudaHipBuiltIn
        , public RoundUniformCudaHipBuiltIn
        , public RsqrtUniformCudaHipBuiltIn
        , public SinUniformCudaHipBuiltIn
        , public SinhUniformCudaHipBuiltIn
        , public SinCosUniformCudaHipBuiltIn
        , public SqrtUniformCudaHipBuiltIn
        , public TanUniformCudaHipBuiltIn
        , public TanhUniformCudaHipBuiltIn
        , public TruncUniformCudaHipBuiltIn
        , public IsnanUniformCudaHipBuiltIn
        , public IsinfUniformCudaHipBuiltIn
        , public IsfiniteUniformCudaHipBuiltIn
    {
    };

    namespace trait
    {
        /* The 2.2 customization entry point for the CUDA/HIP built in implementation.
         *
         * The original 2.2 backend specializations called the CUDA/HIP math intrinsics explicitly.  Here the
         * dispatch forwards to the alpaka 3 math API, which is the same host/device capable entry point the
         * aV2P adapters use: on a CUDA/HIP backend it resolves to the intrinsic implementations and on the host
         * to the standard library, matching the 2.2 `MathUniformCudaHipBuiltIn` behavior.
         */
#define AV2P_CUDAHIP_MATH_TRAIT_UNARY(TMarker, TFunctionName)                                                         \
    template<typename TArg>                                                                                           \
    struct alpakaMathTrait<TMarker, TArg>                                                                             \
    {                                                                                                                 \
        AV2P_FN_HOST_ACC auto operator()(TMarker const& /* ctx */, TArg const& arg) const                             \
        {                                                                                                             \
            return alpaka::math::TFunctionName(arg);                                                                  \
        }                                                                                                             \
    };

#define AV2P_CUDAHIP_MATH_TRAIT_BINARY(TMarker, TFunctionName)                                                        \
    template<typename Tx, typename Ty>                                                                                \
    struct alpakaMathTrait<TMarker, Tx, Ty>                                                                           \
    {                                                                                                                 \
        AV2P_FN_HOST_ACC auto operator()(TMarker const& /* ctx */, Tx const& x, Ty const& y) const                    \
        {                                                                                                             \
            return alpaka::math::TFunctionName(x, y);                                                                 \
        }                                                                                                             \
    };

        AV2P_CUDAHIP_MATH_TRAIT_UNARY(AbsUniformCudaHipBuiltIn, abs)
        AV2P_CUDAHIP_MATH_TRAIT_UNARY(AcosUniformCudaHipBuiltIn, acos)
        AV2P_CUDAHIP_MATH_TRAIT_UNARY(AcoshUniformCudaHipBuiltIn, acosh)
        AV2P_CUDAHIP_MATH_TRAIT_UNARY(ArgUniformCudaHipBuiltIn, arg)
        AV2P_CUDAHIP_MATH_TRAIT_UNARY(AsinUniformCudaHipBuiltIn, asin)
        AV2P_CUDAHIP_MATH_TRAIT_UNARY(AsinhUniformCudaHipBuiltIn, asinh)
        AV2P_CUDAHIP_MATH_TRAIT_UNARY(AtanUniformCudaHipBuiltIn, atan)
        AV2P_CUDAHIP_MATH_TRAIT_UNARY(AtanhUniformCudaHipBuiltIn, atanh)
        AV2P_CUDAHIP_MATH_TRAIT_UNARY(CbrtUniformCudaHipBuiltIn, cbrt)
        AV2P_CUDAHIP_MATH_TRAIT_UNARY(CeilUniformCudaHipBuiltIn, ceil)
        AV2P_CUDAHIP_MATH_TRAIT_UNARY(ConjUniformCudaHipBuiltIn, conj)
        AV2P_CUDAHIP_MATH_TRAIT_UNARY(CosUniformCudaHipBuiltIn, cos)
        AV2P_CUDAHIP_MATH_TRAIT_UNARY(CoshUniformCudaHipBuiltIn, cosh)
        AV2P_CUDAHIP_MATH_TRAIT_UNARY(ErfUniformCudaHipBuiltIn, erf)
        AV2P_CUDAHIP_MATH_TRAIT_UNARY(ExpUniformCudaHipBuiltIn, exp)
        AV2P_CUDAHIP_MATH_TRAIT_UNARY(FloorUniformCudaHipBuiltIn, floor)
        AV2P_CUDAHIP_MATH_TRAIT_UNARY(IsfiniteUniformCudaHipBuiltIn, isfinite)
        AV2P_CUDAHIP_MATH_TRAIT_UNARY(IsinfUniformCudaHipBuiltIn, isinf)
        AV2P_CUDAHIP_MATH_TRAIT_UNARY(IsnanUniformCudaHipBuiltIn, isnan)
        AV2P_CUDAHIP_MATH_TRAIT_UNARY(LogUniformCudaHipBuiltIn, log)
        AV2P_CUDAHIP_MATH_TRAIT_UNARY(Log2UniformCudaHipBuiltIn, log2)
        AV2P_CUDAHIP_MATH_TRAIT_UNARY(Log10UniformCudaHipBuiltIn, log10)
        AV2P_CUDAHIP_MATH_TRAIT_UNARY(RoundUniformCudaHipBuiltIn, round)
        AV2P_CUDAHIP_MATH_TRAIT_UNARY(RsqrtUniformCudaHipBuiltIn, rsqrt)
        AV2P_CUDAHIP_MATH_TRAIT_UNARY(SinUniformCudaHipBuiltIn, sin)
        AV2P_CUDAHIP_MATH_TRAIT_UNARY(SinhUniformCudaHipBuiltIn, sinh)
        AV2P_CUDAHIP_MATH_TRAIT_UNARY(SqrtUniformCudaHipBuiltIn, sqrt)
        AV2P_CUDAHIP_MATH_TRAIT_UNARY(TanUniformCudaHipBuiltIn, tan)
        AV2P_CUDAHIP_MATH_TRAIT_UNARY(TanhUniformCudaHipBuiltIn, tanh)
        AV2P_CUDAHIP_MATH_TRAIT_UNARY(TruncUniformCudaHipBuiltIn, trunc)
        AV2P_CUDAHIP_MATH_TRAIT_BINARY(Atan2UniformCudaHipBuiltIn, atan2)
        AV2P_CUDAHIP_MATH_TRAIT_BINARY(CopysignUniformCudaHipBuiltIn, copysign)
        AV2P_CUDAHIP_MATH_TRAIT_BINARY(FmodUniformCudaHipBuiltIn, fmod)
        AV2P_CUDAHIP_MATH_TRAIT_BINARY(MaxUniformCudaHipBuiltIn, max)
        AV2P_CUDAHIP_MATH_TRAIT_BINARY(MinUniformCudaHipBuiltIn, min)
        AV2P_CUDAHIP_MATH_TRAIT_BINARY(PowUniformCudaHipBuiltIn, pow)
        AV2P_CUDAHIP_MATH_TRAIT_BINARY(RemainderUniformCudaHipBuiltIn, remainder)

        //! The CUDA fma trait specialization.
        template<typename Tx, typename Ty, typename Tz>
        struct alpakaMathTrait<FmaUniformCudaHipBuiltIn, Tx, Ty, Tz>
        {
            AV2P_FN_HOST_ACC auto operator()(
                FmaUniformCudaHipBuiltIn const& /* ctx */,
                Tx const& x,
                Ty const& y,
                Tz const& z) const
            {
                return alpaka::math::fma(x, y, z);
            }
        };

        //! The CUDA sincos trait specialization.
        template<typename TArg>
        struct alpakaMathTrait<SinCosUniformCudaHipBuiltIn, TArg>
        {
            AV2P_FN_HOST_ACC auto operator()(
                SinCosUniformCudaHipBuiltIn const& /* ctx */,
                TArg const& arg,
                TArg& result_sin,
                TArg& result_cos) const -> void
            {
                alpaka::math::sincos(arg, result_sin, result_cos);
            }
        };

#undef AV2P_CUDAHIP_MATH_TRAIT_UNARY
#undef AV2P_CUDAHIP_MATH_TRAIT_BINARY

    } // namespace trait

} // namespace aV2P::math

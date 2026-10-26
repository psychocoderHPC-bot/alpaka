/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

//! Common base for the public (legacy shaped) accelerator class templates.
//!
//! The public accelerators are class templates deriving from `AccAlpaka3` so that user code can use
//! them as deduction patterns (`Acc<TArgs...>`) like the 2.2 class templates.  The device primitive
//! adapters are written against `AccAlpaka3`, so the public type additionally tags itself as an
//! implementation of every accelerator-side concept, each mapping to the `AccAlpaka3` base.  The
//! generic `interface::ImplementationBase` lookup then dispatches every primitive to the existing
//! `AccAlpaka3` specialization, independent of the concrete public wrapper type.

#include "aV2P/atomic/Traits.hpp"
#include "aV2P/block/shared/dyn/Traits.hpp"
#include "aV2P/block/shared/st/Traits.hpp"
#include "aV2P/block/sync/Traits.hpp"
#include "aV2P/core/Interface.hpp"
#include "aV2P/idx/Traits.hpp"
#include "aV2P/intrinsic/Traits.hpp"
#include "aV2P/math/Traits.hpp"
#include "aV2P/mem/fence/Traits.hpp"
#include "aV2P/rand/Traits.hpp"
#include "aV2P/warp/Traits.hpp"
#include "aV2P/workdiv/Traits.hpp"

namespace aV2P::adapter
{
    template<typename TBase>
    class PublicAccFacade
        : public TBase
        , public interface::Implements<ConceptWorkDiv, TBase>
        , public interface::Implements<ConceptIdxGb, TBase>
        , public interface::Implements<ConceptIdxBt, TBase>
        , public interface::Implements<ConceptIntrinsic, TBase>
        , public interface::Implements<ConceptAtomicGrids, TBase>
        , public interface::Implements<ConceptAtomicBlocks, TBase>
        , public interface::Implements<ConceptAtomicThreads, TBase>
        , public interface::Implements<ConceptBlockSync, TBase>
        , public interface::Implements<ConceptBlockSharedDyn, TBase>
        , public interface::Implements<ConceptBlockSharedSt, TBase>
        , public interface::Implements<ConceptMemFence, TBase>
        , public interface::Implements<rand::ConceptRand, TBase>
        , public interface::Implements<warp::ConceptWarp, TBase>
        , public interface::Implements<math::ConceptMathAbs, TBase>
        , public interface::Implements<math::ConceptMathAcos, TBase>
        , public interface::Implements<math::ConceptMathAcosh, TBase>
        , public interface::Implements<math::ConceptMathArg, TBase>
        , public interface::Implements<math::ConceptMathAsin, TBase>
        , public interface::Implements<math::ConceptMathAsinh, TBase>
        , public interface::Implements<math::ConceptMathAtan, TBase>
        , public interface::Implements<math::ConceptMathAtan2, TBase>
        , public interface::Implements<math::ConceptMathAtanh, TBase>
        , public interface::Implements<math::ConceptMathCbrt, TBase>
        , public interface::Implements<math::ConceptMathCeil, TBase>
        , public interface::Implements<math::ConceptMathConj, TBase>
        , public interface::Implements<math::ConceptMathCopysign, TBase>
        , public interface::Implements<math::ConceptMathCos, TBase>
        , public interface::Implements<math::ConceptMathCosh, TBase>
        , public interface::Implements<math::ConceptMathErf, TBase>
        , public interface::Implements<math::ConceptMathExp, TBase>
        , public interface::Implements<math::ConceptMathFloor, TBase>
        , public interface::Implements<math::ConceptMathFma, TBase>
        , public interface::Implements<math::ConceptMathFmod, TBase>
        , public interface::Implements<math::ConceptMathIsfinite, TBase>
        , public interface::Implements<math::ConceptMathIsinf, TBase>
        , public interface::Implements<math::ConceptMathIsnan, TBase>
        , public interface::Implements<math::ConceptMathLog, TBase>
        , public interface::Implements<math::ConceptMathLog10, TBase>
        , public interface::Implements<math::ConceptMathLog2, TBase>
        , public interface::Implements<math::ConceptMathMax, TBase>
        , public interface::Implements<math::ConceptMathMin, TBase>
        , public interface::Implements<math::ConceptMathPow, TBase>
        , public interface::Implements<math::ConceptMathRemainder, TBase>
        , public interface::Implements<math::ConceptMathRound, TBase>
        , public interface::Implements<math::ConceptMathRsqrt, TBase>
        , public interface::Implements<math::ConceptMathSin, TBase>
        , public interface::Implements<math::ConceptMathSinCos, TBase>
        , public interface::Implements<math::ConceptMathSinh, TBase>
        , public interface::Implements<math::ConceptMathSqrt, TBase>
        , public interface::Implements<math::ConceptMathTan, TBase>
        , public interface::Implements<math::ConceptMathTanh, TBase>
        , public interface::Implements<math::ConceptMathTrunc, TBase>
    {
    public:
        using TBase::TBase;
    };
} // namespace aV2P::adapter

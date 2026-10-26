/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Interface.hpp"
#include "aV2P/core/Positioning.hpp"
#include "aV2P/idx/Traits.hpp"
#include "aV2P/vec/Vec.hpp"

namespace aV2P
{
    namespace bt
    {
        //! A zero block thread index provider.
        template<typename TDim, typename TIdx>
        class IdxBtZero : public interface::Implements<ConceptIdxBt, IdxBtZero<TDim, TIdx>>
        {
        };
    } // namespace bt

    namespace trait
    {
        //! The zero block thread index provider dimension get trait specialization.
        template<typename TDim, typename TIdx>
        struct DimType<bt::IdxBtZero<TDim, TIdx>>
        {
            using type = TDim;
        };

        //! The zero block thread index provider block thread index get trait specialization.
        template<typename TDim, typename TIdx>
        struct GetIdx<bt::IdxBtZero<TDim, TIdx>, origin::Block, unit::Threads>
        {
            //! \return The index of the current thread in the block.
            template<typename TWorkDiv>
            AV2P_FN_HOST static auto getIdx(
                bt::IdxBtZero<TDim, TIdx> const& /* idx */,
                TWorkDiv const& /* workDiv */) -> Vec<TDim, TIdx>
            {
                return Vec<TDim, TIdx>::zeros();
            }
        };

        //! The zero block thread index idx type trait specialization.
        template<typename TDim, typename TIdx>
        struct IdxType<bt::IdxBtZero<TDim, TIdx>>
        {
            using type = TIdx;
        };
    } // namespace trait
} // namespace aV2P

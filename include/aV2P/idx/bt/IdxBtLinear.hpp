/* Copyright 2022 Axel Huebl, Jeffrey Kelling, Benjamin Worpitz, René Widera, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Interface.hpp"
#include "aV2P/core/Positioning.hpp"
#include "aV2P/idx/MapIdx.hpp"
#include "aV2P/idx/Traits.hpp"
#include "aV2P/vec/Vec.hpp"
#include "aV2P/workdiv/Traits.hpp"

namespace aV2P
{
    namespace bt
    {
        //! General ND bt index provider based on a linear index.
        template<typename TDim, typename TIdx>
        class IdxBtLinear : public interface::Implements<ConceptIdxBt, IdxBtLinear<TDim, TIdx>>
        {
        public:
            IdxBtLinear(TIdx blockThreadIdx) : m_blockThreadIdx(blockThreadIdx)
            {
            }

            const TIdx m_blockThreadIdx;
        };
    } // namespace bt

    namespace trait
    {
        //! The IdxBtLinear index dimension get trait specialization.
        template<typename TDim, typename TIdx>
        struct DimType<bt::IdxBtLinear<TDim, TIdx>>
        {
            using type = TDim;
        };

        //! The IdxBtLinear block thread index get trait specialization.
        template<typename TDim, typename TIdx>
        struct GetIdx<bt::IdxBtLinear<TDim, TIdx>, origin::Block, unit::Threads>
        {
            //! \return The index of the current thread in the block.
            template<typename TWorkDiv>
            static auto getIdx(bt::IdxBtLinear<TDim, TIdx> const& idx, TWorkDiv const& workDiv) -> Vec<TDim, TIdx>
            {
                return mapIdx<TDim::value>(
                    Vec<DimInt<1u>, TIdx>(idx.m_blockThreadIdx),
                    getWorkDiv<Block, Threads>(workDiv));
            }
        };

        template<typename TIdx>
        struct GetIdx<bt::IdxBtLinear<DimInt<1u>, TIdx>, origin::Block, unit::Threads>
        {
            //! \return The index of the current thread in the block.
            template<typename TWorkDiv>
            static auto getIdx(bt::IdxBtLinear<DimInt<1u>, TIdx> const& idx, TWorkDiv const&) -> Vec<DimInt<1u>, TIdx>
            {
                return idx.m_blockThreadIdx;
            }
        };

        //! The IdxBtLinear block thread index idx type trait specialization.
        template<typename TDim, typename TIdx>
        struct IdxType<bt::IdxBtLinear<TDim, TIdx>>
        {
            using type = TIdx;
        };
    } // namespace trait
} // namespace aV2P

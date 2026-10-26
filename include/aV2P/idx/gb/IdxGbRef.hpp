/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Interface.hpp"
#include "aV2P/core/Positioning.hpp"
#include "aV2P/dim/Traits.hpp"
#include "aV2P/idx/Traits.hpp"
#include "aV2P/vec/Vec.hpp"

namespace aV2P
{
    namespace gb
    {
        //! A IdxGbRef grid block index.
        template<typename TDim, typename TIdx>
        class IdxGbRef : public interface::Implements<ConceptIdxGb, IdxGbRef<TDim, TIdx>>
        {
        public:
            IdxGbRef(Vec<TDim, TIdx> const& gridBlockIdx) : m_gridBlockIdx(gridBlockIdx)
            {
            }

            Vec<TDim, TIdx> const& m_gridBlockIdx;
        };
    } // namespace gb

    namespace trait
    {
        //! The IdxGbRef grid block index dimension get trait specialization.
        template<typename TDim, typename TIdx>
        struct DimType<gb::IdxGbRef<TDim, TIdx>>
        {
            using type = TDim;
        };

        //! The IdxGbRef grid block index grid block index get trait specialization.
        template<typename TDim, typename TIdx>
        struct GetIdx<gb::IdxGbRef<TDim, TIdx>, origin::Grid, unit::Blocks>
        {
            //! \return The index of the current block in the grid.
            template<typename TWorkDiv>
            AV2P_FN_HOST static auto getIdx(gb::IdxGbRef<TDim, TIdx> const& idx, TWorkDiv const& /* workDiv */)
                -> Vec<TDim, TIdx>
            {
                return idx.m_gridBlockIdx;
            }
        };

        //! The IdxGbRef grid block index idx type trait specialization.
        template<typename TDim, typename TIdx>
        struct IdxType<gb::IdxGbRef<TDim, TIdx>>
        {
            using type = TIdx;
        };
    } // namespace trait
} // namespace aV2P

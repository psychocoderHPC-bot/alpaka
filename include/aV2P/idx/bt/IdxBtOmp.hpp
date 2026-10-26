/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Assert.hpp"
#include "aV2P/core/Interface.hpp"
#include "aV2P/core/Positioning.hpp"
#include "aV2P/idx/MapIdx.hpp"
#include "aV2P/idx/Traits.hpp"
#include "aV2P/vec/Vec.hpp"
#include "aV2P/workdiv/Traits.hpp"

#ifdef _OPENMP

#    include <omp.h>

namespace aV2P
{
    namespace bt
    {
        //! The OpenMP accelerator index provider.
        template<typename TDim, typename TIdx>
        class IdxBtOmp : public interface::Implements<ConceptIdxBt, IdxBtOmp<TDim, TIdx>>
        {
        };
    } // namespace bt

    namespace trait
    {
        //! The OpenMP accelerator index dimension get trait specialization.
        template<typename TDim, typename TIdx>
        struct DimType<bt::IdxBtOmp<TDim, TIdx>>
        {
            using type = TDim;
        };

        //! The OpenMP accelerator block thread index get trait specialization.
        template<typename TDim, typename TIdx>
        struct GetIdx<bt::IdxBtOmp<TDim, TIdx>, origin::Block, unit::Threads>
        {
            //! \return The index of the current thread in the block.
            template<typename TWorkDiv>
            static auto getIdx(bt::IdxBtOmp<TDim, TIdx> const& /* idx */, TWorkDiv const& workDiv) -> Vec<TDim, TIdx>
            {
                // We assume that the thread id is positive.
                AV2P_ASSERT_ACC(::omp_get_thread_num() >= 0);
                // \TODO: Would it be faster to precompute the index and cache it inside an array?
                return mapIdx<TDim::value>(
                    Vec<DimInt<1u>, TIdx>(static_cast<TIdx>(::omp_get_thread_num())),
                    getWorkDiv<Block, Threads>(workDiv));
            }
        };

        template<typename TIdx>
        struct GetIdx<bt::IdxBtOmp<DimInt<1u>, TIdx>, origin::Block, unit::Threads>
        {
            //! \return The index of the current thread in the block.
            template<typename TWorkDiv>
            static auto getIdx(bt::IdxBtOmp<DimInt<1u>, TIdx> const& /* idx */, TWorkDiv const&)
                -> Vec<DimInt<1u>, TIdx>
            {
                return Vec<DimInt<1u>, TIdx>(static_cast<TIdx>(::omp_get_thread_num()));
            }
        };

        //! The OpenMP accelerator block thread index idx type trait specialization.
        template<typename TDim, typename TIdx>
        struct IdxType<bt::IdxBtOmp<TDim, TIdx>>
        {
            using type = TIdx;
        };
    } // namespace trait
} // namespace aV2P

#endif

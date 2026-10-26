/* Copyright 2022 Benjamin Worpitz, Matthias Werner, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Common.hpp"
#include "aV2P/extent/Traits.hpp"
#include "aV2P/idx/Traits.hpp"
#include "aV2P/vec/Vec.hpp"
#include "aV2P/workdiv/Traits.hpp"

#include <iosfwd>

namespace aV2P
{
    //! A basic class holding the work division as grid block extent, block thread and thread element extent.
    template<typename TDim, typename TIdx>
    class WorkDivMembers : public interface::Implements<ConceptWorkDiv, WorkDivMembers<TDim, TIdx>>
    {
    public:
        AV2P_FN_HOST_ACC WorkDivMembers() = delete;

        //! Accepts different alpaka vector types and takes the last TDim number of items.
        AV2P_NO_HOST_ACC_WARNING
        template<typename TGridBlockExtent, typename TBlockThreadExtent, typename TThreadElemExtent>
        AV2P_FN_HOST_ACC explicit WorkDivMembers(
            TGridBlockExtent const& gridBlockExtent = TGridBlockExtent(),
            TBlockThreadExtent const& blockThreadExtent = TBlockThreadExtent(),
            TThreadElemExtent const& threadElemExtent = TThreadElemExtent())
            : m_gridBlockExtent(getExtentVecEnd<TDim>(gridBlockExtent))
            , m_blockThreadExtent(getExtentVecEnd<TDim>(blockThreadExtent))
            , m_threadElemExtent(getExtentVecEnd<TDim>(threadElemExtent))
        {
        }

        //! \brief Accepts single specific type and is called without explicit template parameters.
        AV2P_NO_HOST_ACC_WARNING
        AV2P_FN_HOST_ACC WorkDivMembers(
            aV2P::Vec<TDim, TIdx> const& gridBlockExtent,
            aV2P::Vec<TDim, TIdx> const& blockThreadExtent,
            aV2P::Vec<TDim, TIdx> const& elemExtent)
            : m_gridBlockExtent(gridBlockExtent)
            , m_blockThreadExtent(blockThreadExtent)
            , m_threadElemExtent(elemExtent)
        {
        }

        AV2P_NO_HOST_ACC_WARNING
        AV2P_FN_HOST_ACC WorkDivMembers(WorkDivMembers const& other)
            : m_gridBlockExtent(other.m_gridBlockExtent)
            , m_blockThreadExtent(other.m_blockThreadExtent)
            , m_threadElemExtent(other.m_threadElemExtent)
        {
        }

        AV2P_NO_HOST_ACC_WARNING
        template<typename TWorkDiv>
        AV2P_FN_HOST_ACC explicit WorkDivMembers(TWorkDiv const& other)
            : m_gridBlockExtent(subVecEnd<TDim>(getWorkDiv<Grid, Blocks>(other)))
            , m_blockThreadExtent(subVecEnd<TDim>(getWorkDiv<Block, Threads>(other)))
            , m_threadElemExtent(subVecEnd<TDim>(getWorkDiv<Thread, Elems>(other)))
        {
        }

        WorkDivMembers(WorkDivMembers&&) = default;
        auto operator=(WorkDivMembers const&) -> WorkDivMembers& = default;
        auto operator=(WorkDivMembers&&) -> WorkDivMembers& = default;

        AV2P_NO_HOST_ACC_WARNING
        template<typename TWorkDiv>
        AV2P_FN_HOST_ACC auto operator=(TWorkDiv const& other) -> WorkDivMembers<TDim, TIdx>&
        {
            m_gridBlockExtent = subVecEnd<TDim>(getWorkDiv<Grid, Blocks>(other));
            m_blockThreadExtent = subVecEnd<TDim>(getWorkDiv<Block, Threads>(other));
            m_threadElemExtent = subVecEnd<TDim>(getWorkDiv<Thread, Elems>(other));
            return *this;
        }

        AV2P_FN_HOST_ACC friend constexpr auto operator==(WorkDivMembers const& a, WorkDivMembers const& b) -> bool
        {
            return a.m_gridBlockExtent == b.m_gridBlockExtent && a.m_blockThreadExtent == b.m_blockThreadExtent
                   && a.m_threadElemExtent == b.m_threadElemExtent;
        }

        AV2P_FN_HOST friend auto operator<<(std::ostream& os, WorkDivMembers const& workDiv) -> std::ostream&
        {
            return os << "{gridBlockExtent: " << workDiv.m_gridBlockExtent
                      << ", blockThreadExtent: " << workDiv.m_blockThreadExtent
                      << ", threadElemExtent: " << workDiv.m_threadElemExtent << "}";
        }

    public:
        Vec<TDim, TIdx> m_gridBlockExtent;
        Vec<TDim, TIdx> m_blockThreadExtent;
        Vec<TDim, TIdx> m_threadElemExtent;
    };

    //! Deduction guide for the constructor which can be called without explicit template type parameters
    template<typename TDim, typename TIdx>
    WorkDivMembers(
        aV2P::Vec<TDim, TIdx> const& gridBlockExtent,
        aV2P::Vec<TDim, TIdx> const& blockThreadExtent,
        aV2P::Vec<TDim, TIdx> const& elemExtent) -> WorkDivMembers<TDim, TIdx>;

    namespace trait
    {
        //! The WorkDivMembers dimension get trait specialization.
        template<typename TDim, typename TIdx>
        struct DimType<WorkDivMembers<TDim, TIdx>>
        {
            using type = TDim;
        };

        //! The WorkDivMembers idx type trait specialization.
        template<typename TDim, typename TIdx>
        struct IdxType<WorkDivMembers<TDim, TIdx>>
        {
            using type = TIdx;
        };

        //! The WorkDivMembers grid block extent trait specialization.
        template<typename TDim, typename TIdx>
        struct GetWorkDiv<WorkDivMembers<TDim, TIdx>, origin::Grid, unit::Blocks>
        {
            //! \return The number of blocks in each dimension of the grid.
            AV2P_NO_HOST_ACC_WARNING
            AV2P_FN_HOST_ACC static auto getWorkDiv(WorkDivMembers<TDim, TIdx> const& workDiv) -> Vec<TDim, TIdx>
            {
                return workDiv.m_gridBlockExtent;
            }
        };

        //! The WorkDivMembers block thread extent trait specialization.
        template<typename TDim, typename TIdx>
        struct GetWorkDiv<WorkDivMembers<TDim, TIdx>, origin::Block, unit::Threads>
        {
            //! \return The number of threads in each dimension of a block.
            AV2P_NO_HOST_ACC_WARNING
            AV2P_FN_HOST_ACC static auto getWorkDiv(WorkDivMembers<TDim, TIdx> const& workDiv) -> Vec<TDim, TIdx>
            {
                return workDiv.m_blockThreadExtent;
            }
        };

        //! The WorkDivMembers thread element extent trait specialization.
        template<typename TDim, typename TIdx>
        struct GetWorkDiv<WorkDivMembers<TDim, TIdx>, origin::Thread, unit::Elems>
        {
            //! \return The number of elements in each dimension of a thread.
            AV2P_NO_HOST_ACC_WARNING
            AV2P_FN_HOST_ACC static auto getWorkDiv(WorkDivMembers<TDim, TIdx> const& workDiv) -> Vec<TDim, TIdx>
            {
                return workDiv.m_threadElemExtent;
            }
        };
    } // namespace trait
} // namespace aV2P

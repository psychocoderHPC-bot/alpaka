/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Common.hpp"
#include "aV2P/dim/Traits.hpp"
#include "aV2P/idx/Traits.hpp"
#include "aV2P/vec/Traits.hpp"
#include "aV2P/vec/Vec.hpp"
#include "aV2P/workdiv/Traits.hpp"

#include <alpaka/Vec.hpp>
#include <alpaka/onHost/ThreadSpec.hpp>

#include <cstdint>
#include <type_traits>

namespace aV2P
{
    namespace trait
    {
        /** Opt-in grid-block offset for a legacy work division.
         *
         * Legacy alpaka has no offset-aware work division type.  This extension lets a user work-division
         * type describe a partial range: the offset is added to every in-kernel grid-block index.  The
         * default is zero, so ordinary `WorkDivMembers` behaviour is unchanged.
         */
        template<typename TWorkDiv, typename TSfinae = void>
        struct GetGridBlockOffset
        {
            AV2P_FN_HOST_ACC static auto getGridBlockOffset(TWorkDiv const&) -> Vec<Dim<TWorkDiv>, Idx<TWorkDiv>>
            {
                return Vec<Dim<TWorkDiv>, Idx<TWorkDiv>>::zeros();
            }
        };

        /** Opt-in block-thread offset for a legacy work division (default zero). */
        template<typename TWorkDiv, typename TSfinae = void>
        struct GetBlockThreadOffset
        {
            AV2P_FN_HOST_ACC static auto getBlockThreadOffset(TWorkDiv const&) -> Vec<Dim<TWorkDiv>, Idx<TWorkDiv>>
            {
                return Vec<Dim<TWorkDiv>, Idx<TWorkDiv>>::zeros();
            }
        };
    } // namespace trait

    AV2P_NO_HOST_ACC_WARNING
    template<typename TWorkDiv>
    AV2P_FN_HOST_ACC auto getGridBlockOffset(TWorkDiv const& workDiv) -> Vec<Dim<TWorkDiv>, Idx<TWorkDiv>>
    {
        return trait::GetGridBlockOffset<TWorkDiv>::getGridBlockOffset(workDiv);
    }

    AV2P_NO_HOST_ACC_WARNING
    template<typename TWorkDiv>
    AV2P_FN_HOST_ACC auto getBlockThreadOffset(TWorkDiv const& workDiv) -> Vec<Dim<TWorkDiv>, Idx<TWorkDiv>>
    {
        return trait::GetBlockThreadOffset<TWorkDiv>::getBlockThreadOffset(workDiv);
    }
} // namespace aV2P

namespace aV2P::adapter
{
    //! Convert an aV2P vector into the alpaka 3 vector type.
    template<typename TDim, typename TIdx>
    AV2P_FN_HOST_ACC constexpr auto toAlpaka(Vec<TDim, TIdx> const& vec)
    {
        alpaka::Vec<TIdx, static_cast<std::uint32_t>(TDim::value)> result{};
        if constexpr(TDim::value > 0u)
        {
            for(typename TDim::value_type i = 0; i < TDim::value; ++i)
                result[i] = vec[i];
        }
        return result;
    }

    //! Convert an alpaka 3 vector into an aV2P vector.
    template<typename TValue, std::uint32_t TDim>
    AV2P_FN_HOST_ACC constexpr auto fromAlpaka(alpaka::Vec<TValue, TDim> const& vec)
    {
        using Dim = DimInt<TDim>;
        Vec<Dim, TValue> result{};
        if constexpr(TDim > 0u)
        {
            for(std::uint32_t i = 0; i < TDim; ++i)
                result[i] = vec[i];
        }
        return result;
    }

    /** Trivially-copyable work-division payload carried as a kernel argument.
     *
     * alpaka 3 has no thread-element layer and no work-division offsets.  The complete legacy work division is
     * therefore carried *into the device pass* as an ordinary kernel argument and turned into the aV2P acc
     * facade by the kernel wrapper.  The type is intentionally an aggregate of plain `Vec` values: it is
     * trivially copyable, so it satisfies `alpaka::concepts::KernelArg` for every backend.
     */
    template<typename TDim, typename TIdx>
    struct WorkDivPayload
    {
        using Dim = TDim;
        using Idx = TIdx;

        Vec<TDim, TIdx> gridBlockExtent;
        Vec<TDim, TIdx> blockThreadExtent;
        Vec<TDim, TIdx> threadElemExtent;
        Vec<TDim, TIdx> gridBlockOffset;
        Vec<TDim, TIdx> blockThreadOffset;
    };

    template<typename TDim, typename TIdx>
    inline constexpr bool isTriviallyCopyableWorkDivPayload = std::is_trivially_copyable_v<WorkDivPayload<TDim, TIdx>>;

    /** Build the complete legacy work division for one launch.
     *
     * The three extents are read through the legacy `getWorkDiv` interface (so any `TWorkDiv` form works),
     * the offsets through the opt-in offset traits above (zero for ordinary work divisions).
     */
    template<typename TDim, typename TIdx, typename TWorkDiv>
    AV2P_FN_HOST auto makeWorkDivPayload(TWorkDiv const& workDiv) -> WorkDivPayload<TDim, TIdx>
    {
        return WorkDivPayload<TDim, TIdx>{
            aV2P::getWorkDiv<origin::Grid, unit::Blocks>(workDiv),
            aV2P::getWorkDiv<origin::Block, unit::Threads>(workDiv),
            aV2P::getWorkDiv<origin::Thread, unit::Elems>(workDiv),
            aV2P::castVec<TIdx>(subVecEnd<TDim>(aV2P::getGridBlockOffset(workDiv))),
            aV2P::castVec<TIdx>(subVecEnd<TDim>(aV2P::getBlockThreadOffset(workDiv)))};
    }

    //! Translate a legacy work division into the alpaka 3 launch shape.
    //!
    //! The legacy grid-block extent maps to `ThreadSpec::getNumBlocks()` and the legacy
    //! block-thread extent maps to `ThreadSpec::getNumThreads()`.  `ThreadSpec` is used instead of
    //! `FrameSpec` because the legacy API exposes exact block/thread indices and extents, which a
    //! `FrameSpec` does not guarantee.  The legacy thread-element extent and the offsets are carried
    //! in-kernel through `WorkDivPayload` (see `adapter/Kernel.hpp`).
    template<typename TWorkDiv, typename TExecutor>
    AV2P_FN_HOST auto toThreadSpec(TWorkDiv const& workDiv, TExecutor const& executor)
    {
        using Dim = aV2P::Dim<TWorkDiv>;
        using Idx = aV2P::Idx<TWorkDiv>;

        Vec<Dim, Idx> const gridBlockExtent = aV2P::getWorkDiv<origin::Grid, unit::Blocks>(workDiv);
        Vec<Dim, Idx> const blockThreadExtent = aV2P::getWorkDiv<origin::Block, unit::Threads>(workDiv);

        return alpaka::onHost::ThreadSpec{toAlpaka(gridBlockExtent), toAlpaka(blockThreadExtent), executor};
    }

    //! Translate only the block/thread part of a legacy work division into an alpaka 3 `ThreadSpec`.
    template<typename TDim, typename TIdx, typename TExecutor>
    AV2P_FN_HOST auto toThreadSpec(
        Vec<TDim, TIdx> const& gridBlockExtent,
        Vec<TDim, TIdx> const& blockThreadExtent,
        TExecutor const& executor)
    {
        return alpaka::onHost::ThreadSpec{toAlpaka(gridBlockExtent), toAlpaka(blockThreadExtent), executor};
    }
} // namespace aV2P::adapter

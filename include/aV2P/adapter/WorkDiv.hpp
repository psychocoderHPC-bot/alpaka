/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Common.hpp"
#include "aV2P/dim/Traits.hpp"
#include "aV2P/idx/Traits.hpp"
#include "aV2P/vec/Vec.hpp"
#include "aV2P/workdiv/Traits.hpp"

#include <alpaka/Vec.hpp>
#include <alpaka/onHost/ThreadSpec.hpp>

#include <cstdint>

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

    //! Legacy thread-element extent of the kernel invocation currently running on this host thread.
    //!
    //! alpaka 3 has no thread-element layer: a `ThreadSpec` only describes blocks and threads per
    //! block.  The legacy `WorkDivMembers::m_threadElemExtent` is therefore carried out-of-band
    //! and published for the duration of one kernel invocation.  On the host this is safe because
    //! one kernel invocation is executed synchronously on a single host thread.  A device-side
    //! implementation is still required for the GPU backends (Phase 3b).
    template<typename TDim, typename TIdx>
    inline thread_local Vec<TDim, TIdx> g_threadElemExtent = Vec<TDim, TIdx>::ones();

    template<typename TDim, typename TIdx>
    AV2P_FN_HOST_ACC auto currentThreadElemExtent() -> Vec<TDim, TIdx>
    {
        return g_threadElemExtent<TDim, TIdx>;
    }

    //! Publish the legacy thread-element extent while a kernel functor is executing.
    template<typename TDim, typename TIdx>
    class ScopedThreadElemExtent
    {
    public:
        AV2P_FN_ACC explicit ScopedThreadElemExtent(Vec<TDim, TIdx> const& extent)
            : m_old{g_threadElemExtent<TDim, TIdx>}
        {
            g_threadElemExtent<TDim, TIdx> = extent;
        }

        ScopedThreadElemExtent(ScopedThreadElemExtent const&) = delete;
        auto operator=(ScopedThreadElemExtent const&) -> ScopedThreadElemExtent& = delete;

        AV2P_FN_ACC ~ScopedThreadElemExtent()
        {
            g_threadElemExtent<TDim, TIdx> = m_old;
        }

    private:
        Vec<TDim, TIdx> m_old;
    };

    //! Translate a legacy work division into the alpaka 3 launch shape.
    //!
    //! The legacy grid-block extent maps to `ThreadSpec::getNumBlocks()` and the legacy
    //! block-thread extent maps to `ThreadSpec::getNumThreads()`.  `ThreadSpec` is used instead of
    //! `FrameSpec` because the legacy API exposes exact block/thread indices and extents, which a
    //! `FrameSpec` does not guarantee.  The legacy thread-element extent is published separately
    //! through `ScopedThreadElemExtent` (see `adapter/Kernel.hpp`).
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
        return alpaka::onHost::ThreadSpec{
            toAlpaka(gridBlockExtent),
            toAlpaka(blockThreadExtent),
            executor};
    }
} // namespace aV2P::adapter

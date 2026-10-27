/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/adapter/WorkDiv.hpp"
#include "aV2P/core/Common.hpp"
#include "aV2P/dev/Traits.hpp"
#include "aV2P/dim/Traits.hpp"
#include "aV2P/mem/view/Traits.hpp"
#include "aV2P/vec/Traits.hpp"

#include <alpaka/alpaka.hpp>

#include <type_traits>

namespace aV2P::adapter
{
    /*! Bridge an aV2P view or buffer to an alpaka 3 non-owning `View`.
     *
     * The legacy `createView(dev, ptr, extent[, pitch])` and the aV2P view types never own the
     * wrapped allocation.  alpaka 3 can express exactly that: `alpaka::View` stores only a raw
     * pointer, extents, pitches and an alignment.  The explicit-pitch `alpaka::makeView` overload
     * used here preserves the origin pointer and the pitches, so sub-views, external pitched
     * memory and buffer allocations all reach `onHost::memcpy`, `memset` and `fill` unchanged.
     * Host-API buffers carry the contiguous 2.2 pitches, device-API buffers carry the native
     * padded alpaka 3 pitches, and `getAlpaka3()` exposes the same layout, so a native alpaka 3
     * consumer of the buffer handle sees the same element offsets.
     *
     * The returned view does not keep the aV2P view, its device or its allocation alive.  This
     * matches the legacy task lifetime contract: the caller must keep the memory valid until the
     * queue has processed the operation.
     *
     * See alpaka3:include/alpaka/mem/View.hpp (non-owning view, `makeView` overloads at
     * :42/:59/:75/:89) and alpaka3:include/alpaka/mem/MdSpan.hpp.
     */
    template<typename TView>
    AV2P_FN_HOST auto makeAlpakaView(TView&& view)
    {
        auto const apiTag = alpaka::getApi(aV2P::getDev(view).getAlpaka3());
        auto const extents = toAlpaka(aV2P::getExtents(view));
        auto const pitches = toAlpaka(aV2P::getPitchesInBytes(view));
        static_assert(
            std::is_same_v<std::remove_cvref_t<decltype(extents)>, std::remove_cvref_t<decltype(pitches)>>,
            "aV2P adapter: extents and pitches must translate to the same alpaka 3 vector type");
        // Buffers know the alignment their storage really provides; plain views and sub-views do
        // not, so they fall back to the element alignment (sub-view pointer arithmetic can
        // invalidate a stronger claim).
        //
        // The buffer type already encodes the correct promise: a host-API multi-dimensional buffer
        // is a contiguous 2.2 re-view and only advertises the element alignment, while a device-API
        // buffer keeps the native alpaka 3 alignment of its padded allocation.  Forwarding the
        // alignment of `getAlpaka3()` therefore preserves both contracts.
        if constexpr(requires { view.getAlpaka3(); })
            return alpaka::makeView(
                apiTag,
                aV2P::getPtrNative(view),
                extents,
                pitches,
                alpaka::getAlignment(view.getAlpaka3()));
        else
            return alpaka::makeView(apiTag, aV2P::getPtrNative(view), extents, pitches);
    }
} // namespace aV2P::adapter

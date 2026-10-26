/* Copyright 2026 Alexander Matthes, Axel Huebl, Benjamin Worpitz, Andrea Bocci, Jan Stephan, Bernhard Manfred Gruber,
 *                Anton Reinhard, Maria Michailidi
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/adapter/DeviceSpec.hpp"
#include "aV2P/adapter/Buf.hpp"
#include "aV2P/adapter/Platform.hpp"
#include "aV2P/adapter/WorkDiv.hpp"
#include "aV2P/dev/Traits.hpp"
#include "aV2P/mem/buf/Traits.hpp"
#include "aV2P/mem/view/Traits.hpp"

#include <alpaka/alpaka.hpp>

#include <type_traits>
#include <utility>

namespace aV2P::adapter::detail
{
    /*! Re-view an alpaka 3 allocation with the legacy 2.2 contiguous pitches.
     *
     * alpaka 3 host allocations pad the innermost row pitch to the backend SIMD alignment, so a
     * multi-dimensional allocation is larger than the logical extent product and rows are not
     * adjacent.  The 2.2 layout contract is contiguous row-major memory:
     * `pitch[i] == product(extents[i+1:]) * sizeof(TElem)` and `std::data(buf)[i]` spans the whole
     * buffer linearly.
     *
     * This helper keeps the alpaka 3 allocation as the owner (same base pointer, same extents,
     * same alignment) and only replaces the pitches with contiguous ones.  Ownership is retained
     * by capturing the original `SharedBuffer` in the deleter of the new buffer, so the memory is
     * freed exactly once by the original allocator.  No data is copied and nothing is allocated a
     * second time.  The allocated byte range is at least the contiguous span (padding only adds
     * bytes), so every contiguous index stays inside the allocation.
     *
     * Consequence: `getAlpaka3()` returns this re-view, therefore aV2P and a native alpaka 3
     * consumer of the same handle observe identical element offsets.
     */
    template<typename TElem, typename TBuffer>
    AV2P_FN_HOST auto makeContiguous2p2Buffer(TBuffer native)
    {
        using Buffer = std::remove_cvref_t<TBuffer>;
        auto const extents = native.getExtents();
        auto const pitches = aV2P::adapter::toAlpaka(
            aV2P::detail::calculatePitchesFromExtents<TElem>(aV2P::adapter::fromAlpaka(extents)));
        return Buffer{
            alpaka::getApi(native),
            native.data(),
            extents,
            pitches,
            [keepAlive = native]() noexcept {},
            alpaka::getAlignment(native)};
    }
} // namespace aV2P::adapter::detail

namespace aV2P::trait
{
    template<typename TElem, typename TDim, typename TIdx, typename TDevSpec>
    struct BufType<DevAlpaka3<TDevSpec>, TElem, TDim, TIdx>
    {
        using type = BufAlpaka3<TDevSpec, TElem, TDim, TIdx>;
    };

    template<typename TElem, typename TDim, typename TIdx, typename TDevSpec>
    struct ConstBufType<DevAlpaka3<TDevSpec>, TElem, TDim, TIdx>
    {
        using type = ConstBufAlpaka3<TDevSpec, TElem, TDim, TIdx>;
    };

    template<typename TElem, typename TDim, typename TIdx, typename TDevSpec>
    struct DevType<BufAlpaka3<TDevSpec, TElem, TDim, TIdx>>
    {
        using type = DevAlpaka3<TDevSpec>;
    };

    template<typename TElem, typename TDim, typename TIdx, typename TDevSpec>
    struct DevType<ConstBufAlpaka3<TDevSpec, TElem, TDim, TIdx>>
    {
        using type = DevAlpaka3<TDevSpec>;
    };

    template<typename TElem, typename TDim, typename TIdx, typename TDevSpec>
    struct GetDev<BufAlpaka3<TDevSpec, TElem, TDim, TIdx>>
    {
        AV2P_FN_HOST static auto getDev(BufAlpaka3<TDevSpec, TElem, TDim, TIdx> const& buf)
            -> DevAlpaka3<TDevSpec>
        {
            return buf.m_dev;
        }
    };

    template<typename TElem, typename TDim, typename TIdx, typename TDevSpec>
    struct GetDev<ConstBufAlpaka3<TDevSpec, TElem, TDim, TIdx>>
    {
        AV2P_FN_HOST static auto getDev(ConstBufAlpaka3<TDevSpec, TElem, TDim, TIdx> const& buf)
            -> DevAlpaka3<TDevSpec>
        {
            return buf.m_dev;
        }
    };

    template<typename TElem, typename TDim, typename TIdx, typename TDevSpec>
    struct DimType<BufAlpaka3<TDevSpec, TElem, TDim, TIdx>>
    {
        using type = TDim;
    };

    template<typename TElem, typename TDim, typename TIdx, typename TDevSpec>
    struct DimType<ConstBufAlpaka3<TDevSpec, TElem, TDim, TIdx>>
    {
        using type = TDim;
    };

    template<typename TElem, typename TDim, typename TIdx, typename TDevSpec>
    struct ElemType<BufAlpaka3<TDevSpec, TElem, TDim, TIdx>>
    {
        using type = TElem;
    };

    template<typename TElem, typename TDim, typename TIdx, typename TDevSpec>
    struct ElemType<ConstBufAlpaka3<TDevSpec, TElem, TDim, TIdx>>
    {
        // The inner alpaka 3 view carries the const element type.
        using type = TElem const;
    };

    template<typename TElem, typename TDim, typename TIdx, typename TDevSpec>
    struct IdxType<BufAlpaka3<TDevSpec, TElem, TDim, TIdx>>
    {
        using type = TIdx;
    };

    template<typename TElem, typename TDim, typename TIdx, typename TDevSpec>
    struct IdxType<ConstBufAlpaka3<TDevSpec, TElem, TDim, TIdx>>
    {
        using type = TIdx;
    };

    template<typename TElem, typename TDim, typename TIdx, typename TDevSpec>
    struct GetExtents<BufAlpaka3<TDevSpec, TElem, TDim, TIdx>>
    {
        AV2P_FN_HOST auto operator()(BufAlpaka3<TDevSpec, TElem, TDim, TIdx> const& buf) const
        {
            return adapter::fromAlpaka(buf.getAlpaka3().getExtents());
        }
    };

    template<typename TElem, typename TDim, typename TIdx, typename TDevSpec>
    struct GetExtents<ConstBufAlpaka3<TDevSpec, TElem, TDim, TIdx>>
    {
        AV2P_FN_HOST auto operator()(ConstBufAlpaka3<TDevSpec, TElem, TDim, TIdx> const& buf) const
        {
            return adapter::fromAlpaka(buf.getAlpaka3().getExtents());
        }
    };

    //! Report the 2.2 contiguous pitches of the re-viewed allocation.
    //!
    //! `getAlpaka3()` is not the raw padded alpaka 3 allocation: the allocator re-views the same
    //! memory with contiguous pitches (see `adapter::detail::makeContiguous2p2Buffer`).  The
    //! pitches reported here therefore equal `extents * sizeof(TElem)` and match `data()`, the view
    //! accessor, all memory operations and a native alpaka 3 consumer of `getAlpaka3()`.
    template<typename TElem, typename TDim, typename TIdx, typename TDevSpec>
    struct GetPitchesInBytes<BufAlpaka3<TDevSpec, TElem, TDim, TIdx>>
    {
        AV2P_FN_HOST auto operator()(BufAlpaka3<TDevSpec, TElem, TDim, TIdx> const& buf) const
        {
            return aV2P::adapter::fromAlpaka(buf.getAlpaka3().getPitches());
        }
    };

    template<typename TElem, typename TDim, typename TIdx, typename TDevSpec>
    struct GetPitchesInBytes<ConstBufAlpaka3<TDevSpec, TElem, TDim, TIdx>>
    {
        AV2P_FN_HOST auto operator()(ConstBufAlpaka3<TDevSpec, TElem, TDim, TIdx> const& buf) const
        {
            return aV2P::adapter::fromAlpaka(buf.getAlpaka3().getPitches());
        }
    };

    template<typename TElem, typename TDim, typename TIdx, typename TDevSpec>
    struct GetPtrNative<BufAlpaka3<TDevSpec, TElem, TDim, TIdx>>
    {
        AV2P_FN_HOST static auto getPtrNative(BufAlpaka3<TDevSpec, TElem, TDim, TIdx> const& buf) -> TElem const*
        {
            return buf.getAlpaka3().data();
        }

        AV2P_FN_HOST static auto getPtrNative(BufAlpaka3<TDevSpec, TElem, TDim, TIdx>& buf) -> TElem*
        {
            return buf.getAlpaka3().data();
        }
    };

    template<typename TElem, typename TDim, typename TIdx, typename TDevSpec>
    struct GetPtrNative<ConstBufAlpaka3<TDevSpec, TElem, TDim, TIdx>>
    {
        AV2P_FN_HOST static auto getPtrNative(ConstBufAlpaka3<TDevSpec, TElem, TDim, TIdx> const& buf)
            -> TElem const*
        {
            return buf.getAlpaka3().data();
        }
    };

    template<typename TElem, typename TDim, typename TIdx, typename TDevSpec>
    struct GetPtrDev<BufAlpaka3<TDevSpec, TElem, TDim, TIdx>, DevAlpaka3<TDevSpec>>
    {
        AV2P_FN_HOST static auto getPtrDev(
            BufAlpaka3<TDevSpec, TElem, TDim, TIdx> const& buf,
            DevAlpaka3<TDevSpec> const& dev) -> TElem const*
        {
            if(!(dev == getDev(buf)))
                throw std::runtime_error("The buffer is not accessible from the given device!");
            return buf.getAlpaka3().data();
        }

        AV2P_FN_HOST static auto getPtrDev(
            BufAlpaka3<TDevSpec, TElem, TDim, TIdx>& buf,
            DevAlpaka3<TDevSpec> const& dev) -> TElem*
        {
            if(!(dev == getDev(buf)))
                throw std::runtime_error("The buffer is not accessible from the given device!");
            return buf.getAlpaka3().data();
        }
    };

    template<typename TElem, typename TDim, typename TIdx, typename TDevSpec>
    struct GetOffsets<BufAlpaka3<TDevSpec, TElem, TDim, TIdx>>
    {
        AV2P_FN_HOST auto operator()(BufAlpaka3<TDevSpec, TElem, TDim, TIdx> const&) const -> Vec<TDim, TIdx>
        {
            return Vec<TDim, TIdx>::zeros();
        }
    };

    template<typename TElem, typename TDim, typename TIdx, typename TDevSpec>
    struct GetOffsets<ConstBufAlpaka3<TDevSpec, TElem, TDim, TIdx>>
    {
        AV2P_FN_HOST auto operator()(ConstBufAlpaka3<TDevSpec, TElem, TDim, TIdx> const&) const -> Vec<TDim, TIdx>
        {
            return Vec<TDim, TIdx>::zeros();
        }
    };

    template<typename TElem, typename TDim, typename TIdx, typename TDevSpec>
    struct MakeConstBuf<BufAlpaka3<TDevSpec, TElem, TDim, TIdx>>
    {
        AV2P_FN_HOST static auto makeConstBuf(BufAlpaka3<TDevSpec, TElem, TDim, TIdx> const& buf)
            -> ConstBufAlpaka3<TDevSpec, TElem, TDim, TIdx>
        {
            return ConstBufAlpaka3<TDevSpec, TElem, TDim, TIdx>(buf);
        }

        AV2P_FN_HOST static auto makeConstBuf(BufAlpaka3<TDevSpec, TElem, TDim, TIdx>&& buf)
            -> ConstBufAlpaka3<TDevSpec, TElem, TDim, TIdx>
        {
            return ConstBufAlpaka3<TDevSpec, TElem, TDim, TIdx>(std::move(buf));
        }
    };

    template<typename TElem, typename TDim, typename TIdx, typename TDevSpec>
    struct BufAlloc<TElem, TDim, TIdx, DevAlpaka3<TDevSpec>>
    {
        template<typename TExtent>
        AV2P_FN_HOST static auto allocBuf(DevAlpaka3<TDevSpec> const& dev, TExtent const& extent)
            -> BufAlpaka3<TDevSpec, TElem, TDim, TIdx>
        {
            using Buffer = BufAlpaka3<TDevSpec, TElem, TDim, TIdx>;
            if constexpr(std::is_integral_v<TExtent>)
                return Buffer{
                    adapter::detail::makeContiguous2p2Buffer<TElem>(
                        alpaka::onHost::alloc<TElem>(dev.getAlpaka3(), alpaka::Vec{extent})),
                    dev};
            else
                return Buffer{
                    adapter::detail::makeContiguous2p2Buffer<TElem>(
                        alpaka::onHost::alloc<TElem>(dev.getAlpaka3(), adapter::toAlpaka(extent))),
                    dev};
        }
    };

    template<typename TDim, typename TDevSpec>
    struct HasAsyncBufSupport<TDim, DevAlpaka3<TDevSpec>> : std::true_type
    {
    };

    template<typename TElem, typename TDim, typename TIdx, typename TDevSpec>
    struct AsyncBufAlloc<TElem, TDim, TIdx, DevAlpaka3<TDevSpec>>
    {
        template<typename TQueue, typename TExtent>
        AV2P_FN_HOST static auto allocAsyncBuf(TQueue queue, TExtent const& extent)
            -> BufAlpaka3<TDevSpec, TElem, TDim, TIdx>
        {
            using Buffer = BufAlpaka3<TDevSpec, TElem, TDim, TIdx>;
            auto const dev = aV2P::getDev(queue);
            if constexpr(std::is_integral_v<TExtent>)
                return Buffer{
                    adapter::detail::makeContiguous2p2Buffer<TElem>(
                        alpaka::onHost::allocDeferred<TElem>(queue.getAlpaka3(), alpaka::Vec{extent})),
                    dev};
            else
                return Buffer{
                    adapter::detail::makeContiguous2p2Buffer<TElem>(
                        alpaka::onHost::allocDeferred<TElem>(queue.getAlpaka3(), adapter::toAlpaka(extent))),
                    dev};
        }
    };

    //! The legacy CPU platform supports pinned/mapped host memory (it is plain host memory).
    //!
    //! The capability is reported for every alpaka 3 platform spec to keep the legacy
    //! `hasMappedBufSupport` contract.  The allocator below only implements the host platform in
    //! this phase; requesting mapped memory for a different platform produces a compile-time
    //! diagnostic instead of silently returning unpinned or wrongly-owned memory.
    template<typename TDevSpec>
    struct HasMappedBufSupport<PlatformAlpaka3<TDevSpec>> : std::true_type
    {
    };

    template<typename TDevSpec, typename TElem, typename TDim, typename TIdx>
    struct BufAllocMapped<PlatformAlpaka3<TDevSpec>, TElem, TDim, TIdx>
    {
        template<typename TExtent>
        AV2P_FN_HOST static auto allocMappedBuf(
            DevCpu const& host,
            PlatformAlpaka3<TDevSpec> const& /* platform */,
            TExtent const& extent) -> BufAlpaka3<TDevSpec, TElem, TDim, TIdx>
        {
            static_assert(
                std::is_same_v<typename TDevSpec::Api, alpaka::api::Host>,
                "aV2P Phase 3b-3: allocMappedBuf is implemented for the host platform only. "
                "Mapped memory on a real device needs a device-side alpaka 3 allocation and is a "
                "Phase 3b GPU task.");
            using Buffer = BufAlpaka3<TDevSpec, TElem, TDim, TIdx>;
            if constexpr(std::is_integral_v<TExtent>)
                return Buffer{
                    adapter::detail::makeContiguous2p2Buffer<TElem>(
                        alpaka::onHost::allocMapped<TElem>(host.getAlpaka3(), alpaka::Vec{extent})),
                    host};
            else
                return Buffer{
                    adapter::detail::makeContiguous2p2Buffer<TElem>(
                        alpaka::onHost::allocMapped<TElem>(host.getAlpaka3(), adapter::toAlpaka(extent))),
                    host};
        }
    };

    template<typename TDevSpec, typename TElem, typename TDim, typename TIdx>
    struct BufAllocManaged<PlatformAlpaka3<TDevSpec>, TElem, TDim, TIdx>
    {
        template<typename TExtent>
        AV2P_FN_HOST static auto allocManagedBuf(
            DevCpu const& host,
            PlatformAlpaka3<TDevSpec> const& /* platform */,
            TExtent const& extent) -> BufAlpaka3<TDevSpec, TElem, TDim, TIdx>
        {
            static_assert(
                std::is_same_v<typename TDevSpec::Api, alpaka::api::Host>,
                "aV2P Phase 3b-3: allocManagedBuf is implemented for the host platform only. "
                "Unified memory on a real device is a Phase 3b GPU task.");
            using Buffer = BufAlpaka3<TDevSpec, TElem, TDim, TIdx>;
            if constexpr(std::is_integral_v<TExtent>)
                return Buffer{
                    adapter::detail::makeContiguous2p2Buffer<TElem>(
                        alpaka::onHost::allocUnified<TElem>(host.getAlpaka3(), alpaka::Vec{extent})),
                    host};
            else
                return Buffer{
                    adapter::detail::makeContiguous2p2Buffer<TElem>(
                        alpaka::onHost::allocUnified<TElem>(host.getAlpaka3(), adapter::toAlpaka(extent))),
                    host};
        }
    };
} // namespace aV2P::trait

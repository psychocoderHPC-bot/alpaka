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
#include <alpaka/api/util.hpp>

#include <tuple>
#include <type_traits>
#include <utility>

namespace aV2P::adapter::detail
{
    //! Translate a legacy extent (integral or aV2P Vec) to the alpaka 3 vector type.
    template<typename TExtent>
    AV2P_FN_HOST auto toAlpakaExtent(TExtent const& extent)
    {
        if constexpr(std::is_integral_v<TExtent>)
            return alpaka::Vec{extent};
        else
            return adapter::toAlpaka(extent);
    }

    /*! Select the device that owns an allocation of the given platform.
     *
     * The legacy mapped/unified allocators take the host device and the target platform but no
     * device index.  A host platform allocates on the host device passed by the caller (2.2
     * `BufAllocMapped<PlatformCpu>` / `BufAllocManaged<PlatformCpu>` used `allocBuf(host, ...)`);
     * a device platform allocates on device 0 of that platform, which matches the 2.2 behaviour
     * of allocating page-locked/unified memory for the first device of the platform.
     */
    template<typename TDevSpec>
    AV2P_FN_HOST auto platformDevice(DevCpu const& host, PlatformAlpaka3<TDevSpec> const& platform)
    {
        if constexpr(std::is_same_v<typename TDevSpec::Api, alpaka::api::Host>)
            return host;
        else
            return getDevByIdx(platform, 0);
    }

    /*! Re-view a host-API alpaka 3 allocation with the legacy 2.2 contiguous pitches.
     *
     * This helper is used only for `api::Host` allocations.  Device-API buffers keep the native
     * alpaka 3 `SharedBuffer` with its padded row pitches and native alignment.
     *
     * alpaka 3 host allocations pad the innermost row pitch to the backend SIMD alignment, so a
     * multi-dimensional allocation is larger than the logical extent product and rows are not
     * adjacent.  The 2.2 layout contract is contiguous row-major memory:
     * `pitch[i] == product(extents[i+1:]) * sizeof(TElem)` and `std::data(buf)[i]` spans the whole
     * buffer linearly.
     *
     * This helper keeps the alpaka 3 allocation as the owner (same base pointer, same extents) and
     * only replaces the pitches with contiguous ones.  Ownership is retained
     * by capturing the original `SharedBuffer` in the deleter of the new buffer, so the memory is
     * freed exactly once by the original allocator.  No data is copied and nothing is allocated a
     * second time.  The allocated byte range is at least the contiguous span (padding only adds
     * bytes), so every contiguous index stays inside the allocation.
     *
     * Consequence: `getAlpaka3()` returns this re-view, therefore aV2P and a native alpaka 3
     * consumer of the same handle observe identical element offsets.  The alignment metadata is the
     * allocation alignment for a one-dimensional buffer and the element alignment otherwise: the
     * contiguous layout of a multi-dimensional re-view no longer guarantees SIMD-aligned interior
     * rows, and claiming otherwise would let native alpaka 3 bulk operations perform misaligned
     * SIMD accesses.  No padding guarantee is lost because the allocation only grows.
     */
    template<typename TElem, typename TBuffer>
    AV2P_FN_HOST auto makeContiguous2p2Buffer(TBuffer native)
    {
        using Buffer = std::remove_cvref_t<TBuffer>;
        constexpr bool isOneDimensional = Buffer::dim() <= 1u;
        using ContiguousBuffer
            = typename aV2P::detail::NativeBufferWithMatchingAlignment<Buffer, isOneDimensional>::type;
        auto const extents = native.getExtents();
        auto const pitches = aV2P::adapter::toAlpaka(
            aV2P::detail::calculatePitchesFromExtents<TElem>(aV2P::adapter::fromAlpaka(extents)));
        if constexpr(isOneDimensional)
            return ContiguousBuffer{
                alpaka::getApi(native),
                native.data(),
                extents,
                pitches,
                [keepAlive = native]() noexcept {},
                alpaka::getAlignment(native)};
        else
            return ContiguousBuffer{
                alpaka::getApi(native),
                native.data(),
                extents,
                pitches,
                [keepAlive = native]() noexcept {},
                alpaka::Alignment<>{}};
    }

    /*! Re-view a device-API allocation as a host-bound 2.2 buffer.
     *
     * 2.2 returned `BufCpu` from `allocMappedBuf` and `allocManagedBuf` on every backend: both
     * allocations are host-accessible (pinned mapped host memory, or unified/managed memory), and
     * the buffer exposed host element access and reported the host device.  Keep that contract by
     * wrapping the same pointer in a host-API `SharedBuffer` with contiguous 2.2 pitches.  The
     * native allocation travels in the zero-argument deleter, so it is released exactly once by
     * the original allocator.  For multi-dimensional buffers the host buffer type only promises
     * element alignment, matching the contiguous re-view rule.
     */
    template<typename TElem, typename TDim, typename TIdx, typename TNative>
    AV2P_FN_HOST auto makeHostOwning2p2Buffer(TNative native, DevCpu const& host) -> BufCpu<TElem, TDim, TIdx>
    {
        using Buffer = BufCpu<TElem, TDim, TIdx>;
        using NativeBuffer = typename Buffer::AlpakaBufferType;
        constexpr auto alignment = NativeBuffer::getAlignment();
        auto const extents = native.getExtents();
        auto const pitches = alpaka::calculatePitchesFromExtents<TElem>(extents);
        NativeBuffer hostBuffer(
            alpaka::api::Host{},
            native.data(),
            extents,
            pitches,
            [keepAlive = native]() noexcept {},
            alignment);
        return Buffer{hostBuffer, host};
    }

    /*! Build an aV2P buffer around memory obtained from a legacy 2.2 allocator.
     *
     * A legacy allocator only exposes a raw allocation, so there is no native alpaka 3 allocation
     * to preserve.  Reproduce the layout of the matching alpaka 3 allocator:
     *  - Host API: contiguous row-major pitches (`pitch[i] == product(extents[i+1:]) * sizeof(T)`),
     *    exactly like the contiguous host re-view.  Multi-dimensional buffers advertise only the
     *    element alignment, one-dimensional buffers the host allocation alignment.
     *  - Device API: the padded row pitches of `api::util::emulatedAlignedMemDescription` with the
     *    native alignment of the buffer type, so `getAlpaka3()`, the device kernels and the
     *    allocator-backed storage agree on the same pitched offsets.
     */
    template<typename TElem, typename TDim, typename TIdx, typename TDevSpec, typename TAllocator, typename TExtents>
    AV2P_FN_HOST auto makeAllocatorBuffer(
        DevAlpaka3<TDevSpec> const& dev,
        TExtents const& extents,
        TAllocator allocator) -> BufAlpaka3<TDevSpec, TElem, TDim, TIdx>
    {
        using Buffer = BufAlpaka3<TDevSpec, TElem, TDim, TIdx>;
        using NativeBuffer = typename Buffer::AlpakaBufferType;
        constexpr bool isHostApi = std::is_same_v<typename TDevSpec::Api, alpaka::api::Host>;
        // The alpaka 3 allocation type fixes the alignment of the buffer.  Request exactly that
        // alignment from the legacy allocator so the resulting metadata is not stronger than the
        // pointer guarantee.
        constexpr auto alignment = NativeBuffer::getAlignment();
        auto const pitches = [&]()
        {
            if constexpr(isHostApi)
                return alpaka::calculatePitchesFromExtents<TElem>(extents);
            else
                return std::get<1>(alpaka::api::util::emulatedAlignedMemDescription<TElem>(
                    alignment.template get<TElem>(),
                    extents));
        }();
        std::size_t elements;
        if constexpr(isHostApi)
            elements = static_cast<std::size_t>(extents.product());
        else
            elements = static_cast<std::size_t>(pitches[0]) * static_cast<std::size_t>(extents[0]) / sizeof(TElem);
        void* const memPtr = allocator.allocate(elements, alignment.template get<TElem>());
        // alpaka 3 deleters take no arguments; capture the pointer and the allocator.
        auto deleter = [alloc = std::move(allocator), memPtr]() mutable { alloc.deallocate(memPtr); };
        return Buffer{
            NativeBuffer{
                alpaka::getApi(dev.getAlpaka3()),
                static_cast<TElem*>(memPtr),
                extents,
                pitches,
                std::move(deleter),
                alignment},
            dev};
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

    //! Report the pitches of the buffer together with its `getAlpaka3()` view.
    //!
    //! Host-API buffers are re-viewed with contiguous 2.2 pitches (see
    //! `adapter::detail::makeContiguous2p2Buffer`), so the reported pitches equal
    //! `extents * sizeof(TElem)`.  Device-API buffers keep the native alpaka 3 allocation, so the
    //! reported pitches are the native padded row pitches.  In both cases the pitches match
    //! `data()`, the view accessor, all memory operations and a native alpaka 3 consumer of
    //! `getAlpaka3()`.
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
            // Host allocations are re-viewed with the 2.2 contiguous pitches.  Device allocations
            // keep the native alpaka 3 buffer with its padded row pitches and native alignment.
            if constexpr(std::is_same_v<typename TDevSpec::Api, alpaka::api::Host>)
            {
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
            else
            {
                if constexpr(std::is_integral_v<TExtent>)
                    return Buffer{alpaka::onHost::alloc<TElem>(dev.getAlpaka3(), alpaka::Vec{extent}), dev};
                else
                    return Buffer{
                        alpaka::onHost::alloc<TElem>(dev.getAlpaka3(), adapter::toAlpaka(extent)),
                        dev};
            }
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
            if constexpr(std::is_same_v<typename TDevSpec::Api, alpaka::api::Host>)
            {
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
            else
            {
                if constexpr(std::is_integral_v<TExtent>)
                    return Buffer{
                        alpaka::onHost::allocDeferred<TElem>(queue.getAlpaka3(), alpaka::Vec{extent}),
                        dev};
                else
                    return Buffer{
                        alpaka::onHost::allocDeferred<TElem>(
                            queue.getAlpaka3(),
                            adapter::toAlpaka(extent)),
                        dev};
            }
        }
    };

    //! Mapped (pinned, device-visible) host memory support.
    //!
    //! 2.2 reported `true` for `PlatformCpu`, `PlatformUniformCudaHipRt` and
    //! `PlatformGenericSycl`.  Every alpaka 3.0.0 backend this adapter exposes implements
    //! `onHost::allocMapped`:
    //!  - `alpaka/api/host/Device.hpp` (`AllocMapped` forwards to the host allocation),
    //!  - `alpaka/api/unifiedCudaHip/Device.hpp` (`hostMallocMapped | hostMallocPortable`),
    //!  - `alpaka/api/syclGeneric/Device.hpp` (`sycl::aligned_alloc_host`).
    //! Keep the legacy capability constant in sync with those implementations; a backend added
    //! without an `AllocMapped` specialization would fail in `BufAllocMapped` below with a normal
    //! "no matching function for call to allocMapped" diagnostic instead of silently falling back
    //! to non-device-visible memory.
    template<typename TDevSpec>
    struct HasMappedBufSupport<PlatformAlpaka3<TDevSpec>> : std::true_type
    {
    };

    //! Pinned host memory mapped into the address space of the platform's device.
    template<typename TDevSpec, typename TElem, typename TDim, typename TIdx>
    struct BufAllocMapped<PlatformAlpaka3<TDevSpec>, TElem, TDim, TIdx>
    {
        template<typename TExtent>
        AV2P_FN_HOST static auto allocMappedBuf(
            DevCpu const& host,
            PlatformAlpaka3<TDevSpec> const& platform,
            TExtent const& extent) -> BufCpu<TElem, TDim, TIdx>
        {
            auto const dev = adapter::detail::platformDevice<TDevSpec>(host, platform);
            auto const extents = adapter::detail::toAlpakaExtent(extent);
            return adapter::detail::makeHostOwning2p2Buffer<TElem, TDim, TIdx>(
                alpaka::onHost::allocMapped<TElem>(dev.getAlpaka3(), extents),
                host);
        }
    };

    //! Unified/managed memory accessible from the host and the platform's devices.
    template<typename TDevSpec, typename TElem, typename TDim, typename TIdx>
    struct BufAllocManaged<PlatformAlpaka3<TDevSpec>, TElem, TDim, TIdx>
    {
        template<typename TExtent>
        AV2P_FN_HOST static auto allocManagedBuf(
            DevCpu const& host,
            PlatformAlpaka3<TDevSpec> const& platform,
            TExtent const& extent) -> BufCpu<TElem, TDim, TIdx>
        {
            auto const dev = adapter::detail::platformDevice<TDevSpec>(host, platform);
            auto const extents = adapter::detail::toAlpakaExtent(extent);
            return adapter::detail::makeHostOwning2p2Buffer<TElem, TDim, TIdx>(
                alpaka::onHost::allocUnified<TElem>(dev.getAlpaka3(), extents),
                host);
        }
    };

    //! Buffer allocated through a legacy 2.2 `concepts::Allocator`.
    //!
    //! alpaka 3.0.0 has no allocator-aware allocation API.  Construct the buffer from the
    //! allocator's pointer, keep the allocator alive inside the deleter, and use the same
    //! contiguous 2.2 pitches as every other adapter allocation.
    template<typename TElem, typename TDim, typename TIdx, typename TDevSpec, typename TAllocator>
    struct BufAllocWithAllocator<TElem, TDim, TIdx, DevAlpaka3<TDevSpec>, TAllocator>
    {
        template<typename TExtent>
        AV2P_FN_HOST static auto allocBuf(DevAlpaka3<TDevSpec> const& dev, TExtent const& extent, TAllocator allocator)
            -> BufAlpaka3<TDevSpec, TElem, TDim, TIdx>
        {
            return adapter::detail::makeAllocatorBuffer<TElem, TDim, TIdx>(
                dev,
                adapter::detail::toAlpakaExtent(extent),
                std::move(allocator));
        }
    };

    //! Stream-ordered buffer allocated through a legacy 2.2 `concepts::Allocator`.
    //!
    //! 2.2 also allocated this variant synchronously through the allocator and released it in the
    //! buffer deleter; the queue only ordered the user's subsequent work.  Keep that contract.
    template<typename TElem, typename TDim, typename TIdx, typename TDevSpec, typename TAllocator>
    struct AsyncBufAllocWithAllocator<TElem, TDim, TIdx, DevAlpaka3<TDevSpec>, TAllocator>
    {
        template<typename TQueue, typename TExtent>
        AV2P_FN_HOST static auto allocAsyncBuf(TQueue queue, TExtent const& extent, TAllocator allocator)
            -> BufAlpaka3<TDevSpec, TElem, TDim, TIdx>
        {
            return adapter::detail::makeAllocatorBuffer<TElem, TDim, TIdx>(
                aV2P::getDev(queue),
                adapter::detail::toAlpakaExtent(extent),
                std::move(allocator));
        }
    };
} // namespace aV2P::trait

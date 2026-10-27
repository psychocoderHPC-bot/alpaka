/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

//! Zero-dimensional (scalar) memory on top of the one-dimensional aV2P API.
//!
//! Native alpaka 3 rejects zero-dimensional allocations and zero-dimensional views.  This
//! header provides the scalar half of the 2.2 memory interface as a header-only bridge: every
//! scalar is a real one-dimensional, extent-1 allocation, while the wrappers advertise
//! `DimInt<0u>` to aV2P.  aV2P reaches a buffer/view only through `Dim`, `Elem`, `Dev`,
//! `GetExtents`, `GetOffsets`, `GetPitchesInBytes` and `GetPtrNative`, so the wrappers behave like
//! 2.2 zero-dimensional objects and `memcpy`/`memset` copy exactly one element.
//!
//! `aV2P::Buf<TDev, T, DimInt<0u>, Idx>` itself is intentionally not provided: that would require
//! changing the native allocation and view layer.  Use `aV2P::zerodim::ScalarBuf` and
//! `aV2P::zerodim::ScalarView` instead.
//!
//! Allocator selection uses `host_allocator_policy<TDev>` and `device_allocator_policy<TDev>`.
//! Specialise those variables (or `allocCachedBuf`) for a caching allocator.  The defaults are
//! synchronous, so the header works on a bare aV2P install.

#include "aV2P/alpaka.hpp"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <type_traits>
#include <utility>

namespace aV2P::zerodim
{
    using Idx = std::size_t;
    using Dim0D = aV2P::DimInt<0u>;
    using Dim1D = aV2P::DimInt<1u>;
    //! Zero-dimensional extent vector; its product is one.
    using Scalar = aV2P::Vec<Dim0D, Idx>;
    using Ext0 = Scalar;
    using Ext1 = aV2P::Vec<Dim1D, Idx>;

    //! Allocation strategy for the scalar factory functions.
    enum class AllocatorPolicy
    {
        Synchronous = 0,
        Asynchronous = 1,
        Caching = 2
    };

    template<typename TDev>
    inline constexpr AllocatorPolicy host_allocator_policy = AllocatorPolicy::Synchronous;
    template<typename TDev>
    inline constexpr AllocatorPolicy device_allocator_policy = AllocatorPolicy::Synchronous;

    /*! Caching-allocator hook.
     *
     * The default is a plain synchronous allocation.  A consumer with a caching allocator can
     * provide an overload that is found by argument-dependent lookup and set the matching
     * policy variable to `AllocatorPolicy::Caching`.
     */
    template<typename TElem, typename TIdx, typename TDev, typename TQueue, typename TExtent>
    AV2P_FN_HOST auto allocCachedBuf(TDev const& dev, TQueue const&, TExtent const& extent)
    {
        return aV2P::allocBuf<TElem, TIdx>(dev, extent);
    }

    //! Non-owning zero-dimensional view over a single element.
    template<typename TDev, typename T>
    struct ScalarView
    {
        T* m_pMem;
        TDev m_dev;

        AV2P_FN_HOST ScalarView(T* pMem, TDev const& dev) : m_pMem{pMem}, m_dev{dev}
        {
        }

        AV2P_FN_HOST auto data() -> T*
        {
            return m_pMem;
        }

        AV2P_FN_HOST auto data() const -> T const*
        {
            return m_pMem;
        }

        AV2P_FN_HOST auto operator*() -> T&
        {
            return *m_pMem;
        }

        AV2P_FN_HOST auto operator*() const -> T const&
        {
            return *m_pMem;
        }

        AV2P_FN_HOST auto operator->() -> T*
        {
            return m_pMem;
        }

        AV2P_FN_HOST auto operator->() const -> T const*
        {
            return m_pMem;
        }
    };

    //! Owning zero-dimensional buffer: a one-dimensional extent-1 aV2P allocation that advertises
    //! zero-dimensional extents.
    template<typename TDev, typename T>
    struct ScalarBuf
    {
        using Storage = aV2P::Buf<TDev, T, Dim1D, Idx>;

        Storage m_storage;
        T* m_pMem;
        TDev m_dev;

        AV2P_FN_HOST explicit ScalarBuf(Storage&& storage, TDev const& dev)
            : m_storage{std::move(storage)}
            , m_pMem{std::data(m_storage)}
            , m_dev{dev}
        {
        }

        AV2P_FN_HOST auto data() -> T*
        {
            return m_pMem;
        }

        AV2P_FN_HOST auto data() const -> T const*
        {
            return m_pMem;
        }

        AV2P_FN_HOST auto operator*() -> T&
        {
            return *m_pMem;
        }

        AV2P_FN_HOST auto operator*() const -> T const&
        {
            return *m_pMem;
        }

        AV2P_FN_HOST auto operator->() -> T*
        {
            return m_pMem;
        }

        AV2P_FN_HOST auto operator->() const -> T const*
        {
            return m_pMem;
        }
    };

    template<typename TDev, typename T>
    using host_buffer = ScalarBuf<TDev, T>;
    template<typename TDev, typename T>
    using device_buffer = ScalarBuf<TDev, T>;
    template<typename TDev, typename T>
    using host_view = ScalarView<TDev, T>;
    template<typename TDev, typename T>
    using device_view = ScalarView<TDev, T>;

} // namespace aV2P::zerodim

// ---------------------------------------------------------------------------------------------
// aV2P view traits.
// ---------------------------------------------------------------------------------------------
template<typename TDev, typename T>
struct aV2P::trait::DimType<aV2P::zerodim::ScalarView<TDev, T>>
{
    using type = aV2P::DimInt<0u>;
};
template<typename TDev, typename T>
struct aV2P::trait::ElemType<aV2P::zerodim::ScalarView<TDev, T>>
{
    using type = T;
};
template<typename TDev, typename T>
struct aV2P::trait::IdxType<aV2P::zerodim::ScalarView<TDev, T>>
{
    using type = aV2P::zerodim::Idx;
};
template<typename TDev, typename T>
struct aV2P::trait::DevType<aV2P::zerodim::ScalarView<TDev, T>>
{
    using type = aV2P::Dev<TDev>;
};
template<typename TDev, typename T>
struct aV2P::trait::GetDev<aV2P::zerodim::ScalarView<TDev, T>>
{
    AV2P_FN_HOST static auto getDev(aV2P::zerodim::ScalarView<TDev, T> const& view) -> aV2P::Dev<TDev>
    {
        return view.m_dev;
    }
};
template<typename TDev, typename T>
struct aV2P::trait::GetExtents<aV2P::zerodim::ScalarView<TDev, T>>
{
    AV2P_FN_HOST auto operator()(aV2P::zerodim::ScalarView<TDev, T> const&) const -> aV2P::zerodim::Ext0
    {
        return {};
    }
};
template<typename TDev, typename T>
struct aV2P::trait::GetOffsets<aV2P::zerodim::ScalarView<TDev, T>>
{
    AV2P_FN_HOST auto operator()(aV2P::zerodim::ScalarView<TDev, T> const&) const -> aV2P::zerodim::Ext0
    {
        return {};
    }
};
template<typename TDev, typename T>
struct aV2P::trait::GetPtrNative<aV2P::zerodim::ScalarView<TDev, T>>
{
    AV2P_FN_HOST static auto getPtrNative(aV2P::zerodim::ScalarView<TDev, T> const& view) -> T const*
    {
        return view.m_pMem;
    }

    AV2P_FN_HOST static auto getPtrNative(aV2P::zerodim::ScalarView<TDev, T>& view) -> T*
    {
        return view.m_pMem;
    }
};
template<typename TDev, typename T>
struct aV2P::trait::GetPitchesInBytes<aV2P::zerodim::ScalarView<TDev, T>>
{
    AV2P_FN_HOST auto operator()(aV2P::zerodim::ScalarView<TDev, T> const&) const -> aV2P::zerodim::Ext0
    {
        return {};
    }
};

template<typename TDev, typename T>
struct aV2P::trait::DimType<aV2P::zerodim::ScalarBuf<TDev, T>>
{
    using type = aV2P::DimInt<0u>;
};
template<typename TDev, typename T>
struct aV2P::trait::ElemType<aV2P::zerodim::ScalarBuf<TDev, T>>
{
    using type = T;
};
template<typename TDev, typename T>
struct aV2P::trait::IdxType<aV2P::zerodim::ScalarBuf<TDev, T>>
{
    using type = aV2P::zerodim::Idx;
};
template<typename TDev, typename T>
struct aV2P::trait::DevType<aV2P::zerodim::ScalarBuf<TDev, T>>
{
    using type = aV2P::Dev<TDev>;
};
template<typename TDev, typename T>
struct aV2P::trait::GetDev<aV2P::zerodim::ScalarBuf<TDev, T>>
{
    AV2P_FN_HOST static auto getDev(aV2P::zerodim::ScalarBuf<TDev, T> const& view) -> aV2P::Dev<TDev>
    {
        return view.m_dev;
    }
};
template<typename TDev, typename T>
struct aV2P::trait::GetExtents<aV2P::zerodim::ScalarBuf<TDev, T>>
{
    AV2P_FN_HOST auto operator()(aV2P::zerodim::ScalarBuf<TDev, T> const&) const -> aV2P::zerodim::Ext0
    {
        return {};
    }
};
template<typename TDev, typename T>
struct aV2P::trait::GetOffsets<aV2P::zerodim::ScalarBuf<TDev, T>>
{
    AV2P_FN_HOST auto operator()(aV2P::zerodim::ScalarBuf<TDev, T> const&) const -> aV2P::zerodim::Ext0
    {
        return {};
    }
};
template<typename TDev, typename T>
struct aV2P::trait::GetPtrNative<aV2P::zerodim::ScalarBuf<TDev, T>>
{
    AV2P_FN_HOST static auto getPtrNative(aV2P::zerodim::ScalarBuf<TDev, T> const& view) -> T const*
    {
        return view.m_pMem;
    }

    AV2P_FN_HOST static auto getPtrNative(aV2P::zerodim::ScalarBuf<TDev, T>& view) -> T*
    {
        return view.m_pMem;
    }
};
template<typename TDev, typename T>
struct aV2P::trait::GetPitchesInBytes<aV2P::zerodim::ScalarBuf<TDev, T>>
{
    AV2P_FN_HOST auto operator()(aV2P::zerodim::ScalarBuf<TDev, T> const&) const -> aV2P::zerodim::Ext0
    {
        return {};
    }
};

// ---------------------------------------------------------------------------------------------
// Memory operations.  The native aV2P implementations convert every view to a native alpaka 3
// view, which cannot represent zero dimensions.  Translate a 0-D operation to the exactly
// equivalent 1-D extent-1 operation and reuse the native implementation.
// ---------------------------------------------------------------------------------------------
template<typename TDevSpecDst, typename TDevSpecSrc>
struct aV2P::trait::CreateTaskMemcpy<aV2P::DimInt<0u>, aV2P::DevAlpaka3<TDevSpecDst>, aV2P::DevAlpaka3<TDevSpecSrc>>
{
    template<typename TExtent, typename TViewSrc, typename TViewDstFwd>
    AV2P_FN_HOST static auto createTaskMemcpy(TViewDstFwd&& viewDst, TViewSrc const& viewSrc, TExtent const&)
    {
        using Dst = std::remove_reference_t<TViewDstFwd>;
        using Src = TViewSrc;
        using DstDev = aV2P::Dev<Dst>;
        using SrcDev = aV2P::Dev<Src>;
        using DstElem = std::remove_const_t<aV2P::Elem<Dst>>;
        using SrcElem = std::remove_const_t<aV2P::Elem<Src>>;

        aV2P::zerodim::Ext1 const extent1{1u};
        auto* pDst = const_cast<DstElem*>(aV2P::getPtrNative(viewDst));
        auto* pSrc = const_cast<SrcElem*>(aV2P::getPtrNative(viewSrc));
        aV2P::ViewPlainPtr<DstDev, DstElem, aV2P::DimInt<1u>, aV2P::zerodim::Idx> dst1{
            pDst,
            aV2P::getDev(viewDst),
            extent1};
        aV2P::ViewPlainPtr<SrcDev, SrcElem, aV2P::DimInt<1u>, aV2P::zerodim::Idx> src1{
            pSrc,
            aV2P::getDev(viewSrc),
            extent1};
        return aV2P::trait::CreateTaskMemcpy<aV2P::DimInt<1u>, DstDev, SrcDev>::createTaskMemcpy(
            dst1,
            src1,
            extent1);
    }
};

template<typename TDevSpec>
struct aV2P::trait::CreateTaskMemset<aV2P::DimInt<0u>, aV2P::DevAlpaka3<TDevSpec>>
{
    template<typename TExtent, typename TView>
    AV2P_FN_HOST static auto createTaskMemset(TView&& view, std::uint8_t const& byte, TExtent const&)
    {
        using View = std::remove_reference_t<TView>;
        using Dev = aV2P::Dev<View>;
        using Elem = std::remove_const_t<aV2P::Elem<View>>;

        aV2P::zerodim::Ext1 const extent1{1u};
        aV2P::ViewPlainPtr<Dev, Elem, aV2P::DimInt<1u>, aV2P::zerodim::Idx> view1{
            const_cast<Elem*>(aV2P::getPtrNative(view)),
            aV2P::getDev(view),
            extent1};
        return aV2P::trait::CreateTaskMemset<aV2P::DimInt<1u>, Dev>::createTaskMemset(view1, byte, extent1);
    }
};

// ---------------------------------------------------------------------------------------------
// Scalar factory functions, the zero-dimensional counterparts of aV2P::allocBuf and
// aV2P::ViewPlainPtr.  Every allocation is a one-dimensional extent-1 buffer.
// ---------------------------------------------------------------------------------------------
namespace aV2P::zerodim
{
    template<typename T>
    auto make_host_buffer() -> host_buffer<DevCpu, T>
    {
        auto const host = aV2P::getDevByIdx(aV2P::Platform<aV2P::DevCpu>{}, 0u);
        return host_buffer<aV2P::DevCpu, T>{aV2P::allocBuf<T, Idx>(host, Ext1{1u}), host};
    }

    template<typename T, typename TPlatform>
    auto make_host_buffer() -> host_buffer<aV2P::DevCpu, T>
    {
        auto const host = aV2P::getDevByIdx(aV2P::Platform<aV2P::DevCpu>{}, 0u);
        return host_buffer<aV2P::DevCpu, T>{
            aV2P::allocMappedBuf<T, Idx>(host, aV2P::Platform<TPlatform>{}, Ext1{1u}),
            host};
    }

    template<typename T, typename TQueue>
    auto make_host_buffer(TQueue const& queue) -> host_buffer<aV2P::DevCpu, T>
    {
        auto const host = aV2P::getDevByIdx(aV2P::Platform<aV2P::DevCpu>{}, 0u);
        if constexpr(host_allocator_policy<aV2P::DevCpu> == AllocatorPolicy::Caching)
            return host_buffer<aV2P::DevCpu, T>{allocCachedBuf<T, Idx>(host, queue, Ext1{1u}), host};
        else
            return host_buffer<aV2P::DevCpu, T>{
                aV2P::allocMappedBuf<T, Idx>(host, aV2P::Platform<aV2P::DevCpu>{}, Ext1{1u}),
                host};
    }

    template<typename T, typename TDev>
        requires aV2P::concepts::Device<TDev>
    auto make_device_buffer(TDev const& device) -> device_buffer<TDev, T>
    {
        return device_buffer<TDev, T>{aV2P::allocBuf<T, Idx>(device, Ext1{1u}), device};
    }

    template<typename T, typename TQueue>
        requires aV2P::concepts::Queue<TQueue>
    auto make_device_buffer(TQueue const& queue) -> device_buffer<aV2P::Dev<TQueue>, T>
    {
        using TDev = aV2P::Dev<TQueue>;
        auto const dev = aV2P::getDev(queue);
        if constexpr(device_allocator_policy<TDev> == AllocatorPolicy::Caching)
            return device_buffer<TDev, T>{allocCachedBuf<T, Idx>(dev, queue, Ext1{1u}), dev};
        else if constexpr(device_allocator_policy<TDev> == AllocatorPolicy::Asynchronous)
            return device_buffer<TDev, T>{aV2P::allocAsyncBuf<T, Idx>(queue, Ext1{1u}), dev};
        else
            return device_buffer<TDev, T>{aV2P::allocBuf<T, Idx>(dev, Ext1{1u}), dev};
    }

    template<typename TDev, typename T>
        requires aV2P::concepts::Device<TDev>
    auto make_device_view(TDev const& device, T& data) -> device_view<TDev, T>
    {
        return device_view<TDev, T>{&data, device};
    }

    template<typename T, typename TQueue>
        requires aV2P::concepts::Queue<TQueue>
    auto make_device_view(TQueue const& queue, T& data) -> device_view<aV2P::Dev<TQueue>, T>
    {
        return device_view<aV2P::Dev<TQueue>, T>{&data, aV2P::getDev(queue)};
    }

    template<typename T>
    auto make_host_view(T& data) -> host_view<aV2P::DevCpu, T>
    {
        return host_view<aV2P::DevCpu, T>{
            &data,
            aV2P::getDevByIdx(aV2P::Platform<aV2P::DevCpu>{}, 0u)};
    }

} // namespace aV2P::zerodim

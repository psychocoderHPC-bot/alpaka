/* Copyright 2025 Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/acc/Tag.hpp"
#include "aV2P/elem/Traits.hpp"
#include "aV2P/mem/global/Traits.hpp"
#include "aV2P/queue/sycl/QueueGenericSyclBase.hpp"

#ifdef AV2P_ACC_SYCL_ENABLED

#    include <sycl/sycl.hpp>

namespace aV2P
{
    namespace detail
    {
        template<typename T>
        struct DevGlobalTrait<TagCpuSycl, T>
        {
            // SYCL CPU implementation
            using Type = sycl::ext::oneapi::experimental::device_global<T>;
        };

        template<typename T>
        struct DevGlobalTrait<TagGpuSyclIntel, T>
        {
            // SYCL GPU implementation
            using Type = sycl::ext::oneapi::experimental::device_global<T>;
        };

        template<typename T>
        struct DevGlobalTrait<TagGpuSyclNvidia, T>
        {
            // SYCL GPU implementation
            using Type = sycl::ext::oneapi::experimental::device_global<T>;
        };

        template<typename T>
        struct DevGlobalTrait<TagGpuSyclAmd, T>
        {
            // SYCL GPU implementation
            using Type = sycl::ext::oneapi::experimental::device_global<T>;
        };

        template<typename T>
        struct DevGlobalTrait<TagFpgaSyclIntel, T>
        {
            // SYCL FPGA implementation
            using Type = sycl::ext::oneapi::experimental::device_global<T>;
        };

        template<typename T>
        struct DevGlobalTrait<TagGenericSycl, T>
        {
            // generic SYCL implementation
            using Type = sycl::ext::oneapi::experimental::device_global<T>;
        };
    } // namespace detail

    // from device to host
    template<typename TDev, bool TBlocking, typename TViewDst, typename TTypeSrc>
    AV2P_FN_HOST auto memcpy(
        detail::QueueGenericSyclBase<TDev, TBlocking>& queue,
        TViewDst&& viewDst,
        sycl::ext::oneapi::experimental::device_global<TTypeSrc> const& viewSrc)
    {
        queue.getNativeHandle().memcpy(reinterpret_cast<void*>(getPtrNative(viewDst)), viewSrc);
    }

    // from host to device
    template<typename TDev, bool TBlocking, typename TTypeDst, typename TViewSrc>
    AV2P_FN_HOST auto memcpy(
        detail::QueueGenericSyclBase<TDev, TBlocking>& queue,
        sycl::ext::oneapi::experimental::device_global<TTypeDst>& viewDst,
        TViewSrc const& viewSrc)
    {
        queue.getNativeHandle().memcpy(viewDst, reinterpret_cast<void const*>(getPtrNative(viewSrc)));
    }

    // from device to host
    template<typename TDev, bool TBlocking, typename TViewDst, typename TTypeSrc, typename TExtent>
    AV2P_FN_HOST auto memcpy(
        detail::QueueGenericSyclBase<TDev, TBlocking>& queue,
        TViewDst&& viewDst,
        sycl::ext::oneapi::experimental::device_global<TTypeSrc> const& viewSrc,
        TExtent extent)
    {
        using Elem = aV2P::Elem<std::remove_reference_t<TViewDst>>;
        auto size = static_cast<std::size_t>(getHeight(extent)) * static_cast<std::size_t>(getDepth(extent))
                    * static_cast<std::size_t>(getWidth(extent)) * sizeof(Elem);
        queue.getNativeHandle().memcpy(reinterpret_cast<void*>(getPtrNative(viewDst)), viewSrc, size);
    }

    // from host to device
    template<typename TDev, bool TBlocking, typename TTypeDst, typename TViewSrc, typename TExtent>
    AV2P_FN_HOST auto memcpy(
        detail::QueueGenericSyclBase<TDev, TBlocking>& queue,
        sycl::ext::oneapi::experimental::device_global<TTypeDst>& viewDst,
        TViewSrc const& viewSrc,
        TExtent extent)
    {
        using Elem = aV2P::Elem<TViewSrc>;
        auto size = static_cast<std::size_t>(getHeight(extent)) * static_cast<std::size_t>(getDepth(extent))
                    * static_cast<std::size_t>(getWidth(extent)) * sizeof(Elem);
        queue.getNativeHandle().memcpy(viewDst, reinterpret_cast<void const*>(getPtrNative(viewSrc)), size);
    }
} // namespace aV2P
#endif

/* Copyright 2026 Benjamin Worpitz, Erik Zenker, Matthias Werner, René Widera, Andrea Bocci, Jan Stephan,
 *                Bernhard Manfred Gruber, Maria Michailidi
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/adapter/Dev.hpp"
#include "aV2P/adapter/Queue.hpp"
#include "aV2P/adapter/ViewTraits.hpp"
#include "aV2P/adapter/WorkDiv.hpp"
#include "aV2P/core/Common.hpp"
#include "aV2P/dev/Traits.hpp"
#include "aV2P/elem/Traits.hpp"
#include "aV2P/extent/Traits.hpp"
#include "aV2P/mem/view/Traits.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <type_traits>

namespace aV2P
{
    namespace detail
    {
        template<typename TViewDst, typename TViewSrc, typename TExtents>
        struct TaskCopyAlpaka3
        {
            TViewDst m_dest;
            TViewSrc m_src;
            TExtents m_extents;
        };

        template<typename TView, typename TElem, typename TExtents>
        struct TaskFillAlpaka3
        {
            TView m_dest;
            TExtents m_extents;
            TElem m_value;
        };

        template<typename TView, typename TExtents>
        struct TaskSetAlpaka3
        {
            TView m_dest;
            TExtents m_extents;
            std::uint8_t m_value;
        };
    } // namespace detail

    namespace trait
    {
        template<typename TDim, typename TDevSpecDst, typename TDevSpecSrc>
        struct CreateTaskMemcpy<TDim, DevAlpaka3<TDevSpecDst>, DevAlpaka3<TDevSpecSrc>>
        {
            template<typename TExtent, typename TViewSrc, typename TViewDstFwd>
            AV2P_FN_HOST static auto createTaskMemcpy(
                TViewDstFwd&& viewDst,
                TViewSrc const& viewSrc,
                TExtent const& extent)
            {
                auto dst = adapter::makeAlpakaView(viewDst);
                auto src = adapter::makeAlpakaView(viewSrc);
                if constexpr(std::is_integral_v<TExtent>)
                    return aV2P::detail::TaskCopyAlpaka3{std::move(dst), std::move(src), alpaka::Vec{extent}};
                else
                    return aV2P::detail::TaskCopyAlpaka3{std::move(dst), std::move(src), adapter::toAlpaka(extent)};
            }
        };

        template<
            typename TDev,
            alpaka::concepts::QueueKind TQueueKind,
            typename TViewDst,
            typename TViewSrc,
            typename TExtent>
        struct Enqueue<QueueAlpaka3<TDev, TQueueKind>, aV2P::detail::TaskCopyAlpaka3<TViewDst, TViewSrc, TExtent>>
        {
            AV2P_FN_HOST static auto enqueue(
                QueueAlpaka3<TDev, TQueueKind>& queue,
                aV2P::detail::TaskCopyAlpaka3<TViewDst, TViewSrc, TExtent> task) -> void
            {
                alpaka::onHost::memcpy(queue.getAlpaka3(), task.m_dest, task.m_src, task.m_extents);
            }
        };

        template<typename TDim, typename TDevSpec>
        struct CreateTaskFill<TDim, DevAlpaka3<TDevSpec>>
        {
            template<typename TExtent, typename TView>
            AV2P_FN_HOST static auto createTaskFill(
                TView&& viewDst,
                aV2P::Elem<std::remove_reference_t<TView>> const& value,
                TExtent const& extent)
            {
                auto dst = adapter::makeAlpakaView(viewDst);
                if constexpr(std::is_integral_v<TExtent>)
                    return aV2P::detail::TaskFillAlpaka3{std::move(dst), alpaka::Vec{extent}, value};
                else
                    return aV2P::detail::TaskFillAlpaka3{std::move(dst), adapter::toAlpaka(extent), value};
            }
        };

        template<
            typename TDev,
            alpaka::concepts::QueueKind TQueueKind,
            typename TView,
            typename TElem,
            typename TExtent>
        struct Enqueue<QueueAlpaka3<TDev, TQueueKind>, aV2P::detail::TaskFillAlpaka3<TView, TElem, TExtent>>
        {
            AV2P_FN_HOST static auto enqueue(
                QueueAlpaka3<TDev, TQueueKind>& queue,
                aV2P::detail::TaskFillAlpaka3<TView, TElem, TExtent> task) -> void
            {
                alpaka::onHost::fill(queue.getAlpaka3(), task.m_dest, task.m_value, task.m_extents);
            }
        };

        template<typename TDim, typename TDevSpec>
        struct CreateTaskMemset<TDim, DevAlpaka3<TDevSpec>>
        {
            template<typename TExtent, typename TView>
            AV2P_FN_HOST static auto createTaskMemset(TView&& viewDst, std::uint8_t const& byte, TExtent const& extent)
            {
                auto dst = adapter::makeAlpakaView(viewDst);
                if constexpr(std::is_integral_v<TExtent>)
                    return aV2P::detail::TaskSetAlpaka3{std::move(dst), alpaka::Vec{extent}, byte};
                else
                    return aV2P::detail::TaskSetAlpaka3{std::move(dst), adapter::toAlpaka(extent), byte};
            }
        };

        template<typename TDev, alpaka::concepts::QueueKind TQueueKind, typename TView, typename TExtent>
        struct Enqueue<QueueAlpaka3<TDev, TQueueKind>, aV2P::detail::TaskSetAlpaka3<TView, TExtent>>
        {
            AV2P_FN_HOST static auto enqueue(
                QueueAlpaka3<TDev, TQueueKind>& queue,
                aV2P::detail::TaskSetAlpaka3<TView, TExtent> task) -> void
            {
                alpaka::onHost::memset(queue.getAlpaka3(), task.m_dest, task.m_value, task.m_extents);
            }
        };

        /*! Legacy native queue handle on the device backends that exposed one in 2.2.
         *
         * 2.2 implemented `trait::NativeHandle` only for the CUDA/HIP stream queue and the SYCL
         * queue (`alpaka/queue/cuda-hip/QueueUniformCudaHipRt.hpp:258`,
         * `alpaka/queue/sycl/QueueGenericSyclBase.hpp:284`); the host queue had no native handle.
         * Keep exactly that set: the CUDA caching-allocator test and other legacy consumers pass
         * the handle to native APIs.  Host queues still fall through to the primary template's
         * "This type does not have a native handle!" diagnostic instead of a fabricated value.
         */
#if defined(AV2P_ACC_GPU_CUDA_ENABLED)
        template<alpaka::concepts::QueueKind TQueueKind>
        struct NativeHandle<QueueAlpaka3<DevCudaRt, TQueueKind>>
        {
            [[nodiscard]] static auto getNativeHandle(QueueAlpaka3<DevCudaRt, TQueueKind> const& queue)
            {
                return queue.getAlpaka3().getNativeHandle();
            }
        };
#endif
#if defined(AV2P_ACC_GPU_HIP_ENABLED)
        template<alpaka::concepts::QueueKind TQueueKind>
        struct NativeHandle<QueueAlpaka3<DevHipRt, TQueueKind>>
        {
            [[nodiscard]] static auto getNativeHandle(QueueAlpaka3<DevHipRt, TQueueKind> const& queue)
            {
                return queue.getAlpaka3().getNativeHandle();
            }
        };
#endif
#if defined(AV2P_ACC_SYCL_ENABLED)
        template<alpaka::concepts::QueueKind TQueueKind>
        struct NativeHandle<QueueAlpaka3<DevCpuSycl, TQueueKind>>
        {
            [[nodiscard]] static auto getNativeHandle(QueueAlpaka3<DevCpuSycl, TQueueKind> const& queue)
            {
                return queue.getAlpaka3().getNativeHandle();
            }
        };

        template<alpaka::concepts::QueueKind TQueueKind>
        struct NativeHandle<QueueAlpaka3<DevGpuSyclIntel, TQueueKind>>
        {
            [[nodiscard]] static auto getNativeHandle(QueueAlpaka3<DevGpuSyclIntel, TQueueKind> const& queue)
            {
                return queue.getAlpaka3().getNativeHandle();
            }
        };
#endif
    } // namespace trait
} // namespace aV2P

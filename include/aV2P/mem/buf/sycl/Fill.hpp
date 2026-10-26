/* Copyright 2025 Maria Michailidi, Anna Polova, Abdulrahman Al Marzouqi
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Debug.hpp"
#include "aV2P/core/Sycl.hpp"
#include "aV2P/dev/DevGenericSycl.hpp"
#include "aV2P/dev/Traits.hpp"
#include "aV2P/dim/DimIntegralConst.hpp"
#include "aV2P/extent/Traits.hpp"
#include "aV2P/mem/buf/sycl/Common.hpp"
#include "aV2P/mem/view/Traits.hpp"
#include "aV2P/meta/NdLoop.hpp"
#include "aV2P/queue/QueueGenericSyclBlocking.hpp"
#include "aV2P/queue/QueueGenericSyclNonBlocking.hpp"
#include "aV2P/queue/Traits.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <type_traits>


#ifdef AV2P_ACC_SYCL_ENABLED

namespace aV2P
{

    namespace detail
    {

        template<typename TDim, typename TView, typename TExtent, typename TValue>
        struct TaskFillSyclBase
        {
            using ExtentSize = Idx<TExtent>;
            using DstSize = Idx<TView>;
            using Elem = aV2P::Elem<TView>;

            template<typename TViewFwd>
            TaskFillSyclBase(TViewFwd&& view, TValue const& value, TExtent const& extent)
                : m_value(value)
                , m_extent(getExtents(extent))
                , m_extentWidth(m_extent.back())
#    if(!defined(NDEBUG)) || (AV2P_DEBUG >= AV2P_DEBUG_FULL)
                , m_dstExtent(getExtents(view))
#    endif
                , m_dstPitchBytes(getPitchesInBytes(view))
                , m_dstMemNative(getPtrNative(view))
            {
                AV2P_ASSERT((castVec<DstSize>(m_extent) <= m_dstExtent).all());
                if constexpr(TDim::value > 1)
                    AV2P_ASSERT(
                        m_extentWidth * static_cast<ExtentSize>(sizeof(Elem)) <= m_dstPitchBytes[TDim::value - 2]);
            }

#    if AV2P_DEBUG >= AV2P_DEBUG_FULL
            auto printDebug() const -> void
            {
                std::cout << __func__ << " e: " << m_extent << " ew: " << m_extentWidth << " de: " << m_dstExtent
                          << " dptr: " << reinterpret_cast<void*>(m_dstMemNative) << " dpitchb: " << m_dstPitchBytes
                          << std::endl;
            }
#    endif

            TValue const m_value;
            Vec<TDim, ExtentSize> const m_extent;
            ExtentSize const m_extentWidth;
#    if(!defined(NDEBUG)) || (AV2P_DEBUG >= AV2P_DEBUG_FULL)
            Vec<TDim, DstSize> const m_dstExtent;
#    endif
            Vec<TDim, DstSize> const m_dstPitchBytes;
            Elem* const m_dstMemNative;

            static constexpr auto is_sycl_task = true;
        };

        template<typename TDim, typename TView, typename TExtent, typename TValue>
        struct TaskFillSycl : public TaskFillSyclBase<TDim, TView, TExtent, TValue>
        {
            using Base = TaskFillSyclBase<TDim, TView, TExtent, TValue>;
            using Base::Base;
            using typename Base::DstSize;
            using typename Base::ExtentSize;
            using DimMin1 = DimInt<TDim::value - 1u>;

            auto operator()(sycl::queue& queue, std::vector<sycl::event> const& requirements) const -> sycl::event
            {
                AV2P_DEBUG_MINIMAL_LOG_SCOPE;
#    if AV2P_DEBUG >= AV2P_DEBUG_FULL
                this->printDebug();
#    endif
                Vec<DimMin1, ExtentSize> const extentWithoutInnermost(subVecBegin<DimMin1>(this->m_extent));
                Vec<DimMin1, DstSize> const dstPitchBytesWithoutInnermost(subVecBegin<DimMin1>(this->m_dstPitchBytes));

                std::vector<sycl::event> events;
                events.reserve(static_cast<std::size_t>(extentWithoutInnermost.prod()));

                if(static_cast<std::size_t>(this->m_extent.prod()) != 0u)
                {
                    using Elem = std::remove_cvref_t<decltype(this->m_value)>;

                    meta::ndLoopIncIdx(
                        extentWithoutInnermost,
                        [&](Vec<DimMin1, ExtentSize> const& idx)
                        {
                            auto offsetBytes = (castVec<DstSize>(idx) * dstPitchBytesWithoutInnermost).sum();
                            Elem* ptr = reinterpret_cast<Elem*>(
                                reinterpret_cast<std::uint8_t*>(this->m_dstMemNative) + offsetBytes);

                            assert(this->m_extentWidth >= 0);

                            events.push_back(queue.fill<TValue>(
                                ptr,
                                this->m_value,
                                static_cast<std::size_t>(this->m_extentWidth),
                                requirements));
                        });
                }


                return queue.ext_oneapi_submit_barrier(events);
            }
        };

        template<typename TView, typename TExtent, typename TValue>
        struct TaskFillSycl<DimInt<1u>, TView, TExtent, TValue>
            : public TaskFillSyclBase<DimInt<1u>, TView, TExtent, TValue>
        {
            using Base = TaskFillSyclBase<DimInt<1u>, TView, TExtent, TValue>;
            using Base::Base;

            auto operator()(sycl::queue& queue, std::vector<sycl::event> const& requirements) const -> sycl::event
            {
                AV2P_DEBUG_MINIMAL_LOG_SCOPE;
#    if AV2P_DEBUG >= AV2P_DEBUG_FULL
                this->printDebug();
#    endif
                if(static_cast<std::size_t>(this->m_extent.prod()) != 0u)
                {
                    return queue.fill(
                        this->m_dstMemNative,
                        this->m_value,
                        static_cast<std::size_t>(this->m_extentWidth),
                        requirements);
                }
                else
                {
                    return queue.ext_oneapi_submit_barrier();
                }
            }
        };

        template<typename TView, typename TExtent, typename TValue>
        struct TaskFillSycl<DimInt<0u>, TView, TExtent, TValue>
        {
            using Elem = aV2P::Elem<TView>;

            template<typename TViewFwd>
            TaskFillSycl(TViewFwd&& view, TValue const& value, [[maybe_unused]] TExtent const& extent)
                : m_value(value)
                , m_dstMemNative(getPtrNative(view))
            {
                AV2P_ASSERT(getExtents(extent).prod() == 1u);
                AV2P_ASSERT(getExtents(view).prod() == 1u);
            }

            auto operator()(sycl::queue& queue, std::vector<sycl::event> const& requirements) const -> sycl::event
            {
                AV2P_DEBUG_MINIMAL_LOG_SCOPE;
                return queue.fill(m_dstMemNative, m_value, 1, requirements);
            }

            TValue const m_value;
            Elem* const m_dstMemNative;
            static constexpr auto is_sycl_task = true;
        };

    } // namespace detail

    namespace trait
    {
        template<typename TDim, typename TPlatform>
        struct CreateTaskFill<TDim, DevGenericSycl<TPlatform>>
        {
            template<typename TExtent, typename TView, typename TValue>
            static auto createTaskFill(TView& view, TValue const& value, TExtent const& extent)
                -> aV2P::detail::TaskFillSycl<TDim, TView, TExtent, TValue>
            {
                return aV2P::detail::TaskFillSycl<TDim, TView, TExtent, TValue>(view, value, extent);
            }
        };
    } // namespace trait

} // namespace aV2P
#endif

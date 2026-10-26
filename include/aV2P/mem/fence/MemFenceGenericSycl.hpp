/* Copyright 2023 Jan Stephan, Luca Ferragina, Andrea Bocci, Tapish Narwal
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/mem/fence/Traits.hpp"
#include "aV2P/mem/order/MemoryOrderGenericSycl.hpp"

#ifdef AV2P_ACC_SYCL_ENABLED

#    include <sycl/sycl.hpp>

namespace aV2P
{
    namespace detail
    {
        template<typename TAlpakaMemScope>
        struct SyclFenceProps
        {
        };

        template<>
        struct SyclFenceProps<aV2P::memory_scope::Block>
        {
            static constexpr auto scope = sycl::memory_scope::work_group;
        };

        template<>
        struct SyclFenceProps<aV2P::memory_scope::Device>
        {
            static constexpr auto scope = sycl::memory_scope::device;
        };

        template<>
        struct SyclFenceProps<aV2P::memory_scope::Grid>
        {
            static constexpr auto scope = sycl::memory_scope::device;
        };
    } // namespace detail

    //! The SYCL memory fence.
    class MemFenceGenericSycl : public interface::Implements<ConceptMemFence, MemFenceGenericSycl>
    {
    };
} // namespace aV2P

namespace aV2P::trait
{
    template<>
    struct MemFenceDefaultOrder<MemFenceGenericSycl>
    {
        using type = mem_order::AcqRel;
        static constexpr auto value = mem_order::acq_rel;
    };

    template<MemoryOrder TMemOrder, MemoryScope TMemScope>
    struct MemFence<MemFenceGenericSycl, TMemOrder, TMemScope>
    {
        static auto mem_fence(MemFenceGenericSycl const&, TMemOrder order, TMemScope const&)
        {
            static constexpr auto scope = aV2P::detail::SyclFenceProps<TMemScope>::scope;
            sycl::atomic_fence(MemOrderSycl::get(order), scope);
        }
    };
} // namespace aV2P::trait

#endif

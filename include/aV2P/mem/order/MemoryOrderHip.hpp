/* Copyright 2025 Tapish Narwal
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/mem/order/MemoryOrder.hpp"

#include <concepts>

#ifdef AV2P_ACC_GPU_HIP_ENABLED

namespace aV2P
{
    struct MemOrderHip
    {
        template<MemoryOrder TMemOrder>
        static constexpr auto get(TMemOrder)
        {
            if constexpr(std::same_as<TMemOrder, mem_order::SeqCst>)
            {
                return __ATOMIC_SEQ_CST;
            }
            if constexpr(std::same_as<TMemOrder, mem_order::AcqRel>)
            {
                return __ATOMIC_ACQ_REL;
            }
            if constexpr(std::same_as<TMemOrder, mem_order::Release>)
            {
                return __ATOMIC_RELEASE;
            }
            if constexpr(std::same_as<TMemOrder, mem_order::Acquire>)
            {
                return __ATOMIC_ACQUIRE;
            }
            if constexpr(std::same_as<TMemOrder, mem_order::Relaxed>)
            {
                return __ATOMIC_RELAXED;
            }
        }
    };

} // namespace aV2P

#endif

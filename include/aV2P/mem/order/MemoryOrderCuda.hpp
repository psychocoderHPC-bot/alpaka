/* Copyright 2025 Tapish Narwal
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Config.hpp"
#include "aV2P/core/PP.hpp"
#include "aV2P/mem/order/MemoryOrder.hpp"

#include <concepts>

#ifdef AV2P_ACC_GPU_CUDA_ENABLED
namespace aV2P
{
    struct MemOrderCuda
    {
        template<MemoryOrder TMemOrder>
        static constexpr auto get(TMemOrder)
        {
#    ifdef AV2P_CUDA_ATOMIC
            if constexpr(std::same_as<TMemOrder, mem_order::SeqCst>)
            {
                return ::cuda::memory_order_seq_cst;
            }
            if constexpr(std::same_as<TMemOrder, mem_order::AcqRel>)
            {
                return ::cuda::memory_order_acq_rel;
            }
            if constexpr(std::same_as<TMemOrder, mem_order::Release>)
            {
                return ::cuda::memory_order_release;
            }
            if constexpr(std::same_as<TMemOrder, mem_order::Acquire>)
            {
                return ::cuda::memory_order_acquire;
            }
            if constexpr(std::same_as<TMemOrder, mem_order::Relaxed>)
            {
                return ::cuda::memory_order_relaxed;
            }
#    else
#        if AV2P_LANG_CUDA >= AV2P_VERSION_NUMBER(12, 8, 0) && AV2P_ARCH_PTX

            if constexpr(std::same_as<TMemOrder, mem_order::SeqCst>)
            {
                return __NV_ATOMIC_SEQ_CST;
            }
            if constexpr(std::same_as<TMemOrder, mem_order::AcqRel>)
            {
                return __NV_ATOMIC_ACQ_REL;
            }
            if constexpr(std::same_as<TMemOrder, mem_order::Release>)
            {
                return __NV_ATOMIC_RELEASE;
            }
            if constexpr(std::same_as<TMemOrder, mem_order::Acquire>)
            {
                return __NV_ATOMIC_ACQUIRE;
            }
            if constexpr(std::same_as<TMemOrder, mem_order::Relaxed>)
            {
                return __NV_ATOMIC_RELAXED;
            }
#        endif
#    endif
        }
    };

} // namespace aV2P

#endif

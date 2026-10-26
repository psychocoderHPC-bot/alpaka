/* Copyright 2022 Jan Stephan, Andrea Bocci, Bernhard Manfred Gruber, Tapish Narwal
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Config.hpp"
#include "aV2P/core/Interface.hpp"
#include "aV2P/core/PP.hpp"
#include "aV2P/mem/fence/Traits.hpp"
#include "aV2P/mem/order/MemoryOrder.hpp"
#include "aV2P/mem/order/MemoryOrderCuda.hpp"
#include "aV2P/mem/order/MemoryOrderHip.hpp"

#if defined(AV2P_ACC_GPU_CUDA_ENABLED) || defined(AV2P_ACC_GPU_HIP_ENABLED)

namespace aV2P
{
    //! The GPU CUDA/HIP memory fence.
    class MemFenceUniformCudaHipBuiltIn : public interface::Implements<ConceptMemFence, MemFenceUniformCudaHipBuiltIn>
    {
    };

#    if !defined(AV2P_HOST_ONLY)
#        if defined(AV2P_ACC_GPU_CUDA_ENABLED) && !AV2P_LANG_CUDA
#            error If AV2P_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#        endif

#        if defined(AV2P_ACC_GPU_HIP_ENABLED) && !AV2P_LANG_HIP
#            error If AV2P_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#        endif


    namespace detail
    {
        // For CUDA > 12.8 the compiler inbuilt __nv_atomic_thread_fence is available for compute
        // capability versions > 7. NVCC defines __CUDACC_DEVICE_ATOMIC_BUILTINS__ when built-in atomic functions are
        // supported by the compute capability. Im not sure how this will work with clang-cuda. Currently the inline
        // ptx version of the code is suffiecient for thread fences.
        template<aV2P::MemoryOrder TMemOrder>
        [[maybe_unused]] static constexpr __device__ void cuda_ptx_fence_device([[maybe_unused]] TMemOrder order)
        {
#        if AV2P_ARCH_PTX >= AV2P_VERSION_NUMBER(9, 0, 0)
            // full acquire/release semantics support
            if constexpr(std::is_same_v<TMemOrder, mem_order::Relaxed>)
            { // Relaxed ordering requires no fence
            }
            else if constexpr(std::is_same_v<TMemOrder, mem_order::Acquire>)
            {
                asm volatile("fence.acquire.gpu;" ::);
            }
            else if constexpr(std::is_same_v<TMemOrder, mem_order::Release>)
            {
                asm volatile("fence.release.gpu;" ::);
            }
            else if constexpr(std::is_same_v<TMemOrder, mem_order::AcqRel>)
            {
                asm volatile("fence.acq_rel.gpu;" ::);
            }
            else
            { // Sequential consistency
                asm volatile("fence.sc.gpu;" ::);
            }
#        elif AV2P_ARCH_PTX >= AV2P_VERSION_NUMBER(7, 0, 0)
            // only acq_rel and sc available
            if constexpr(std::is_same_v<TMemOrder, mem_order::Relaxed>)
            { // Relaxed ordering requires no fence
            }
            else if constexpr(std::is_same_v<TMemOrder, mem_order::Acquire>)
            {
                asm volatile("fence.acq_rel.gpu;" ::);
            }
            else if constexpr(std::is_same_v<TMemOrder, mem_order::Release>)
            {
                asm volatile("fence.acq_rel.gpu;" ::);
            }
            else if constexpr(std::is_same_v<TMemOrder, mem_order::AcqRel>)
            {
                asm volatile("fence.acq_rel.gpu;" ::);
            }
            else
            {
                // Sequential consistency
                asm volatile("fence.sc.gpu;" ::);
            }
#        endif
        }

        template<aV2P::MemoryOrder TMemOrder>
        [[maybe_unused]] static constexpr __device__ void cuda_ptx_fence_block([[maybe_unused]] TMemOrder order)
        {
#        if AV2P_ARCH_PTX >= AV2P_VERSION_NUMBER(9, 0, 0)
            // full acquire/release semantics support
            if constexpr(std::is_same_v<TMemOrder, mem_order::Relaxed>)
            { // Relaxed ordering requires no fence
            }
            else if constexpr(std::is_same_v<TMemOrder, mem_order::Acquire>)
            {
                asm volatile("fence.acquire.cta;" ::);
            }
            else if constexpr(std::is_same_v<TMemOrder, mem_order::Release>)
            {
                asm volatile("fence.release.cta;" ::);
            }
            else if constexpr(std::is_same_v<TMemOrder, mem_order::AcqRel>)
            {
                asm volatile("fence.acq_rel.cta;" ::);
            }
            else
            { // Sequential consistency
                asm volatile("fence.sc.cta;" ::);
            }
#        elif AV2P_ARCH_PTX >= AV2P_VERSION_NUMBER(7, 0, 0)
            // only acq_rel and sc available
            if constexpr(std::is_same_v<TMemOrder, mem_order::Relaxed>)
            { // Relaxed ordering requires no fence
            }
            else if constexpr(std::is_same_v<TMemOrder, mem_order::Acquire>)
            {
                asm volatile("fence.acq_rel.cta;" ::);
            }
            else if constexpr(std::is_same_v<TMemOrder, mem_order::Release>)
            {
                asm volatile("fence.acq_rel.cta;" ::);
            }
            else if constexpr(std::is_same_v<TMemOrder, mem_order::AcqRel>)
            {
                asm volatile("fence.acq_rel.cta;" ::);
            }
            else
            { // Sequential consistency
                asm volatile("fence.sc.cta;" ::);
            }
#        endif
        }

        template<aV2P::MemoryOrder TMemOrder>
        [[maybe_unused]] static constexpr __device__ void cuda_mem_fence_block([[maybe_unused]] TMemOrder order)
        {
            if constexpr(std::is_same_v<TMemOrder, mem_order::Relaxed>)
            { // Relaxed ordering requires no fence
                return;
            }
#        ifdef AV2P_CUDA_ATOMIC
            ::cuda::atomic_thread_fence(MemOrderCuda::get(order), ::cuda::thread_scope_block);
#        else
#            if AV2P_ARCH_PTX
#                if AV2P_ARCH_PTX >= AV2P_VERSION_NUMBER(7, 0, 0)
            cuda_ptx_fence_block(order);
#                else
            __threadfence_block();
#                endif
#            endif
#        endif
        }

        template<aV2P::MemoryOrder TMemOrder>
        [[maybe_unused]] static constexpr __device__ void cuda_mem_fence_device([[maybe_unused]] TMemOrder order)
        {
            if constexpr(std::is_same_v<TMemOrder, mem_order::Relaxed>)
            { // Relaxed ordering requires no fence
                return;
            }
#        ifdef AV2P_CUDA_ATOMIC
            ::cuda::atomic_thread_fence(MemOrderCuda::get(order), ::cuda::thread_scope_device);
#        else
#            if AV2P_ARCH_PTX
#                if AV2P_ARCH_PTX >= AV2P_VERSION_NUMBER(7, 0, 0)
            cuda_ptx_fence_device(order);
#                else
            __threadfence();
#                endif
#            endif
#        endif
        }
    } // namespace detail

    namespace trait
    {
        template<>
        struct MemFenceDefaultOrder<MemFenceUniformCudaHipBuiltIn>
        {
            using type = mem_order::SeqCst;
            static constexpr auto value = mem_order::seq_cst;
        };

        template<MemoryOrder TMemOrder>
        struct MemFence<MemFenceUniformCudaHipBuiltIn, TMemOrder, memory_scope::Block>
        {
            static __device__ auto mem_fence(
                MemFenceUniformCudaHipBuiltIn const&,
                TMemOrder order,
                memory_scope::Block const&)
            {
#        ifdef AV2P_ACC_GPU_CUDA_ENABLED
                aV2P::detail::cuda_mem_fence_block(order);
#        else
                /* It is required to have this temporary variable else HIP 7.14 will fail to compile with code
                 * emitting issues.
                 */
                constexpr auto memOrder = MemOrderHip::get(order);
                __builtin_amdgcn_fence(memOrder, "workgroup");
#        endif
            }
        };

        template<MemoryOrder TMemOrder, typename TMemScope>
        struct MemFence<MemFenceUniformCudaHipBuiltIn, TMemOrder, TMemScope>
        {
            static __device__ auto mem_fence(MemFenceUniformCudaHipBuiltIn const&, TMemOrder order, TMemScope const&)
            {
                // Base case for grid and device scope fences.
                // CUDA and HIP do not have a per-grid memory fence, so a device-level fence is used
#        ifdef AV2P_ACC_GPU_CUDA_ENABLED
                aV2P::detail::cuda_mem_fence_device(order);
#        else
                /* It is required to have this temporary variable else HIP 7.14 will fail to compile with code
                 * emitting issues.
                 */
                constexpr auto memOrder = MemOrderHip::get(order);
                __builtin_amdgcn_fence(memOrder, "agent");
#        endif
            }
        };

    } // namespace trait
#    endif

} // namespace aV2P

#endif

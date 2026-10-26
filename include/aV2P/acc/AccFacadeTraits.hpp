/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

//! Trait bridge for the public (legacy shaped) accelerator class templates.
//!
//! The public accelerators derive from `AccAlpaka3`.  The device primitive adapters are specialized
//! on `AccAlpaka3` and reach it through `interface::ImplementationBase` (see
//! `aV2P/acc/AccFacadeBase.hpp`).  The traits below are not dispatched through
//! `ImplementationBase`, so each public wrapper needs an explicit forwarding specialization.  The
//! launch trait is re-implemented instead of forwarded because the in-kernel facade has to be the
//! public wrapper: kernels with a concrete (non-template) accelerator parameter receive that exact
//! type.

#include "aV2P/adapter/Acc.hpp"
#include "aV2P/adapter/AccTraits.hpp"

#include <cstdint>
#include <stdexcept>
#include <utility>

namespace aV2P::adapter::detail
{
    /** Create a legacy kernel task whose in-kernel facade is the public accelerator wrapper. */
    template<
        typename TFacade,
        typename TDim,
        typename TIdx,
        typename TWorkDiv,
        typename TKernelFnObj,
        typename... TArgs>
    struct PublicAccCreateTaskKernel
    {
        AV2P_FN_HOST static auto createTaskKernel(
            TWorkDiv const& workDiv,
            TKernelFnObj const& kernelFnObj,
            TArgs&&... args)
        {
            using Acc = TFacade;

            auto const payload = adapter::makeWorkDivPayload<TDim, TIdx>(workDiv);

            // Both host executors run one thread per block (CpuOmpBlocks parallelises blocks, not
            // threads).  Keep the legacy 2.2 diagnostic instead of letting the backend throw later.
            if constexpr(alpaka::exec::isSeqExecutor_v<typename TFacade::Executor>)
            {
                if(payload.blockThreadExtent.prod() != static_cast<TIdx>(1u))
                {
                    throw std::runtime_error(
                        "The given work division is not valid for this accelerator: threads per block "
                        "must be 1. Use a different accelerator or a work division with one thread per "
                        "block.");
                }
            }

            auto const dynSharedMemBytes = static_cast<std::uint32_t>(
                trait::BlockSharedMemDynSizeBytes<TKernelFnObj, Acc>::getBlockSharedMemDynSizeBytes(
                    kernelFnObj,
                    payload.blockThreadExtent,
                    payload.threadElemExtent,
                    args...));

            adapter::LegacyKernel<Acc, TDim, TIdx, TKernelFnObj> wrapper{kernelFnObj, dynSharedMemBytes};

            return TaskKernelAlpaka3<Acc, TDim, TIdx, decltype(wrapper), TArgs...>(
                workDiv,
                wrapper,
                payload,
                std::forward<TArgs>(args)...);
        }
    };
} // namespace aV2P::adapter::detail

// The mapping tags are single tokens, so they can be passed through the macro below.
#define AV2P_PUBLIC_ACC_BASE(TDim, TIdx, TAG) ::aV2P::adapter::LegacyAcc<TDim, TIdx, TAG>

#define AV2P_DEFINE_PUBLIC_ACC_TRAITS(ACC, TAG)                                                                      \
    namespace aV2P::trait                                                                                            \
    {                                                                                                                \
        template<typename TDim, typename TIdx>                                                                       \
        struct AccType<ACC<TDim, TIdx>>                                                                              \
        {                                                                                                            \
            using type = ACC<TDim, TIdx>;                                                                            \
        };                                                                                                           \
                                                                                                                     \
        template<typename TDim, typename TIdx>                                                                       \
        struct DimType<ACC<TDim, TIdx>> : DimType<AV2P_PUBLIC_ACC_BASE(TDim, TIdx, TAG)>                             \
        {                                                                                                            \
        };                                                                                                           \
                                                                                                                     \
        template<typename TDim, typename TIdx>                                                                       \
        struct IdxType<ACC<TDim, TIdx>> : IdxType<AV2P_PUBLIC_ACC_BASE(TDim, TIdx, TAG)>                             \
        {                                                                                                            \
        };                                                                                                           \
                                                                                                                     \
        template<typename TDim, typename TIdx>                                                                       \
        struct DevType<ACC<TDim, TIdx>> : DevType<AV2P_PUBLIC_ACC_BASE(TDim, TIdx, TAG)>                             \
        {                                                                                                            \
        };                                                                                                           \
                                                                                                                     \
        template<typename TDim, typename TIdx>                                                                       \
        struct PlatformType<ACC<TDim, TIdx>> : PlatformType<AV2P_PUBLIC_ACC_BASE(TDim, TIdx, TAG)>                   \
        {                                                                                                            \
        };                                                                                                           \
                                                                                                                     \
        template<typename TDim, typename TIdx>                                                                       \
        struct GetAccName<ACC<TDim, TIdx>> : GetAccName<AV2P_PUBLIC_ACC_BASE(TDim, TIdx, TAG)>                       \
        {                                                                                                            \
        };                                                                                                           \
                                                                                                                     \
        template<typename TDim, typename TIdx>                                                                       \
        struct GetAccDevProps<ACC<TDim, TIdx>> : GetAccDevProps<AV2P_PUBLIC_ACC_BASE(TDim, TIdx, TAG)>                \
        {                                                                                                            \
        };                                                                                                           \
                                                                                                                     \
        template<typename TDim, typename TIdx>                                                                       \
        struct IsSingleThreadAcc<ACC<TDim, TIdx>>                                                                    \
            : IsSingleThreadAcc<AV2P_PUBLIC_ACC_BASE(TDim, TIdx, TAG)>                                               \
        {                                                                                                            \
        };                                                                                                           \
                                                                                                                     \
        template<typename TDim, typename TIdx>                                                                       \
        struct IsMultiThreadAcc<ACC<TDim, TIdx>>                                                                     \
            : IsMultiThreadAcc<AV2P_PUBLIC_ACC_BASE(TDim, TIdx, TAG)>                                                \
        {                                                                                                            \
        };                                                                                                           \
                                                                                                                     \
        template<typename TDim, typename TIdx, typename TDev, typename TKernelFnObj, typename... TArgs>              \
        struct FunctionAttributes<ACC<TDim, TIdx>, TDev, TKernelFnObj, TArgs...>                                     \
            : FunctionAttributes<AV2P_PUBLIC_ACC_BASE(TDim, TIdx, TAG), TDev, TKernelFnObj, TArgs...>                \
        {                                                                                                            \
        };                                                                                                           \
                                                                                                                     \
        template<typename TDim, typename TIdx, typename TWorkDiv, typename TKernelFnObj, typename... TArgs>          \
        struct CreateTaskKernel<ACC<TDim, TIdx>, TWorkDiv, TKernelFnObj, TArgs...>                                   \
            : ::aV2P::adapter::detail::PublicAccCreateTaskKernel<                                                    \
                  ACC<TDim, TIdx>,                                                                                   \
                  TDim,                                                                                              \
                  TIdx,                                                                                              \
                  TWorkDiv,                                                                                          \
                  TKernelFnObj,                                                                                      \
                  TArgs...>                                                                                          \
        {                                                                                                            \
        };                                                                                                           \
                                                                                                                     \
        template<                                                                                                    \
            typename TDim,                                                                                           \
            typename TIdx,                                                                                           \
            typename TDev,                                                                                           \
            typename TKernelFnObj,                                                                                   \
            typename TWorkDim,                                                                                       \
            typename TWorkIdx,                                                                                       \
            typename... TArgs>                                                                                       \
        struct MaxActiveBlocks<ACC<TDim, TIdx>, TDev, TKernelFnObj, TWorkDim, TWorkIdx, TArgs...>                    \
            : MaxActiveBlocks<AV2P_PUBLIC_ACC_BASE(TDim, TIdx, TAG), TDev, TKernelFnObj, TWorkDim, TWorkIdx, TArgs...>\
        {                                                                                                            \
        };                                                                                                           \
                                                                                                                     \
        template<typename TDim, typename TIdx, typename TWorkDiv, typename TKernelFnObj, typename... TArgs>          \
        struct CreateTaskCooperativeKernel<ACC<TDim, TIdx>, TWorkDiv, TKernelFnObj, TArgs...>                        \
            : CreateTaskCooperativeKernel<                                                                           \
                  AV2P_PUBLIC_ACC_BASE(TDim, TIdx, TAG),                                                             \
                  TWorkDiv,                                                                                          \
                  TKernelFnObj,                                                                                      \
                  TArgs...>                                                                                          \
        {                                                                                                            \
        };                                                                                                           \
    }

#ifdef AV2P_ACC_CPU_B_SEQ_T_SEQ_ENABLED
AV2P_DEFINE_PUBLIC_ACC_TRAITS(::aV2P::AccCpuSerial, ::aV2P::TagCpuSerial);
#endif

#ifdef AV2P_ACC_CPU_B_OMP2_T_SEQ_ENABLED
AV2P_DEFINE_PUBLIC_ACC_TRAITS(::aV2P::AccCpuOmp2Blocks, ::aV2P::TagCpuOmp2Blocks);
#endif

#ifdef AV2P_ACC_GPU_CUDA_ENABLED
AV2P_DEFINE_PUBLIC_ACC_TRAITS(::aV2P::AccGpuCudaRt, ::aV2P::TagGpuCudaRt);
#endif

#ifdef AV2P_ACC_GPU_HIP_ENABLED
AV2P_DEFINE_PUBLIC_ACC_TRAITS(::aV2P::AccGpuHipRt, ::aV2P::TagGpuHipRt);
#endif

#if defined(AV2P_ACC_SYCL_ENABLED) && defined(AV2P_SYCL_TARGET_CPU)
AV2P_DEFINE_PUBLIC_ACC_TRAITS(::aV2P::AccCpuSycl, ::aV2P::TagCpuSycl);
#endif

#if defined(AV2P_ACC_SYCL_ENABLED) && defined(AV2P_SYCL_TARGET_GPU)
AV2P_DEFINE_PUBLIC_ACC_TRAITS(::aV2P::AccGpuSyclIntel, ::aV2P::TagGpuSyclIntel);
#endif

#undef AV2P_DEFINE_PUBLIC_ACC_TRAITS
#undef AV2P_PUBLIC_ACC_BASE

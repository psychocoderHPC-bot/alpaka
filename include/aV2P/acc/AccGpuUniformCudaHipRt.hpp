/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

//! Legacy accelerator name kept for source compatibility; the implementation is the alpaka 3 adapter.
//!
//! The first (now unused) template parameter is the legacy API tag; it is kept so that existing
//! code can spell the three-parameter legacy name.  The second API tag path maps to the corresponding
//! accelerator of the enabled GPU backend.

#include "aV2P/adapter/Acc.hpp"
#include "aV2P/acc/AccFacadeTraits.hpp"

namespace aV2P
{
#if defined(AV2P_ACC_GPU_CUDA_ENABLED)
    template<typename TApi, typename TDim, typename TIdx>
    class AccGpuUniformCudaHipRt : public AccGpuCudaRt<TDim, TIdx>
    {
        using Base = AccGpuCudaRt<TDim, TIdx>;

    public:
        using Base::Base;
    };

    namespace trait
    {
        template<typename TApi, typename TDim, typename TIdx>
        struct AccType<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>>
        {
            using type = AccGpuUniformCudaHipRt<TApi, TDim, TIdx>;
        };

        template<typename TApi, typename TDim, typename TIdx>
        struct DimType<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>> : DimType<AccGpuCudaRt<TDim, TIdx>>
        {
        };

        template<typename TApi, typename TDim, typename TIdx>
        struct IdxType<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>> : IdxType<AccGpuCudaRt<TDim, TIdx>>
        {
        };

        template<typename TApi, typename TDim, typename TIdx>
        struct DevType<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>> : DevType<AccGpuCudaRt<TDim, TIdx>>
        {
        };

        template<typename TApi, typename TDim, typename TIdx>
        struct PlatformType<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>>
            : PlatformType<AccGpuCudaRt<TDim, TIdx>>
        {
        };

        template<typename TApi, typename TDim, typename TIdx>
        struct GetAccName<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>> : GetAccName<AccGpuCudaRt<TDim, TIdx>>
        {
        };

        template<typename TApi, typename TDim, typename TIdx>
        struct IsSingleThreadAcc<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>>
            : IsSingleThreadAcc<AccGpuCudaRt<TDim, TIdx>>
        {
        };

        template<typename TApi, typename TDim, typename TIdx>
        struct IsMultiThreadAcc<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>>
            : IsMultiThreadAcc<AccGpuCudaRt<TDim, TIdx>>
        {
        };

        template<
            typename TApi,
            typename TDim,
            typename TIdx,
            typename TDev,
            typename TKernelFnObj,
            typename... TArgs>
        struct FunctionAttributes<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>, TDev, TKernelFnObj, TArgs...>
            : FunctionAttributes<AccGpuCudaRt<TDim, TIdx>, TDev, TKernelFnObj, TArgs...>
        {
        };

        template<
            typename TApi,
            typename TDim,
            typename TIdx,
            typename TWorkDiv,
            typename TKernelFnObj,
            typename... TArgs>
        struct CreateTaskKernel<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>, TWorkDiv, TKernelFnObj, TArgs...>
            : adapter::detail::PublicAccCreateTaskKernel<
                  AccGpuUniformCudaHipRt<TApi, TDim, TIdx>,
                  TDim,
                  TIdx,
                  TWorkDiv,
                  TKernelFnObj,
                  TArgs...>
        {
        };
    } // namespace trait
#elif defined(AV2P_ACC_GPU_HIP_ENABLED)
    template<typename TApi, typename TDim, typename TIdx>
    class AccGpuUniformCudaHipRt : public AccGpuHipRt<TDim, TIdx>
    {
        using Base = AccGpuHipRt<TDim, TIdx>;

    public:
        using Base::Base;
    };

    namespace trait
    {
        template<typename TApi, typename TDim, typename TIdx>
        struct AccType<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>>
        {
            using type = AccGpuUniformCudaHipRt<TApi, TDim, TIdx>;
        };

        template<typename TApi, typename TDim, typename TIdx>
        struct DimType<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>> : DimType<AccGpuHipRt<TDim, TIdx>>
        {
        };

        template<typename TApi, typename TDim, typename TIdx>
        struct IdxType<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>> : IdxType<AccGpuHipRt<TDim, TIdx>>
        {
        };

        template<typename TApi, typename TDim, typename TIdx>
        struct DevType<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>> : DevType<AccGpuHipRt<TDim, TIdx>>
        {
        };

        template<typename TApi, typename TDim, typename TIdx>
        struct PlatformType<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>>
            : PlatformType<AccGpuHipRt<TDim, TIdx>>
        {
        };

        template<typename TApi, typename TDim, typename TIdx>
        struct GetAccName<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>> : GetAccName<AccGpuHipRt<TDim, TIdx>>
        {
        };

        template<typename TApi, typename TDim, typename TIdx>
        struct IsSingleThreadAcc<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>>
            : IsSingleThreadAcc<AccGpuHipRt<TDim, TIdx>>
        {
        };

        template<typename TApi, typename TDim, typename TIdx>
        struct IsMultiThreadAcc<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>>
            : IsMultiThreadAcc<AccGpuHipRt<TDim, TIdx>>
        {
        };

        template<
            typename TApi,
            typename TDim,
            typename TIdx,
            typename TDev,
            typename TKernelFnObj,
            typename... TArgs>
        struct FunctionAttributes<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>, TDev, TKernelFnObj, TArgs...>
            : FunctionAttributes<AccGpuHipRt<TDim, TIdx>, TDev, TKernelFnObj, TArgs...>
        {
        };

        template<
            typename TApi,
            typename TDim,
            typename TIdx,
            typename TWorkDiv,
            typename TKernelFnObj,
            typename... TArgs>
        struct CreateTaskKernel<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>, TWorkDiv, TKernelFnObj, TArgs...>
            : adapter::detail::PublicAccCreateTaskKernel<
                  AccGpuUniformCudaHipRt<TApi, TDim, TIdx>,
                  TDim,
                  TIdx,
                  TWorkDiv,
                  TKernelFnObj,
                  TArgs...>
        {
        };
    } // namespace trait
#endif
} // namespace aV2P

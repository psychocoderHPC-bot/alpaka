/* Copyright 2026 Benjamin Worpitz, René Widera, Jan Stephan, Andrea Bocci, Bernhard Manfred Gruber, Antonio Di Pilato
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// Base classes.
#include "aV2P/atomic/AtomicHierarchy.hpp"
#include "aV2P/atomic/AtomicUniformCudaHipBuiltIn.hpp"
#include "aV2P/block/shared/dyn/BlockSharedMemDynUniformCudaHipBuiltIn.hpp"
#include "aV2P/block/shared/st/BlockSharedMemStUniformCudaHipBuiltIn.hpp"
#include "aV2P/block/sync/BlockSyncUniformCudaHipBuiltIn.hpp"
#include "aV2P/core/DemangleTypeNames.hpp"
#include "aV2P/grid/GridSyncGpuCudaHip.hpp"
#include "aV2P/idx/bt/IdxBtUniformCudaHipBuiltIn.hpp"
#include "aV2P/idx/gb/IdxGbUniformCudaHipBuiltIn.hpp"
#include "aV2P/intrinsic/IntrinsicUniformCudaHipBuiltIn.hpp"
#include "aV2P/math/MathUniformCudaHipBuiltIn.hpp"
#include "aV2P/mem/fence/MemFenceUniformCudaHipBuiltIn.hpp"
#include "aV2P/rand/RandDefault.hpp"
#include "aV2P/rand/RandUniformCudaHipRand.hpp"
#include "aV2P/warp/WarpUniformCudaHipBuiltIn.hpp"
#include "aV2P/workdiv/WorkDivUniformCudaHipBuiltIn.hpp"

// Specialized traits.
#include "aV2P/acc/Traits.hpp"
#include "aV2P/dev/Traits.hpp"
#include "aV2P/idx/Traits.hpp"
#include "aV2P/kernel/Traits.hpp"
#include "aV2P/platform/Traits.hpp"

// Implementation details.
#include "aV2P/core/ClipCast.hpp"
#include "aV2P/core/Cuda.hpp"
#include "aV2P/core/Interface.hpp"
#include "aV2P/dev/DevUniformCudaHipRt.hpp"

#ifdef __cpp_lib_format
#    include <format>
#endif
#include <string>

#if defined(AV2P_ACC_GPU_CUDA_ENABLED) || defined(AV2P_ACC_GPU_HIP_ENABLED)

namespace aV2P
{
    template<
        typename TApi,
        typename TAcc,
        typename TDim,
        typename TIdx,
        typename TKernelFnObj,
        bool TCooperative,
        typename... TArgs>
    class TaskKernelGpuUniformCudaHipRt;

    //! The GPU CUDA accelerator.
    //!
    //! This accelerator allows parallel kernel execution on devices supporting CUDA.
    template<typename TApi, typename TDim, typename TIdx>
    class AccGpuUniformCudaHipRt final
        : public WorkDivUniformCudaHipBuiltIn<TDim, TIdx>
        , public gb::IdxGbUniformCudaHipBuiltIn<TDim, TIdx>
        , public bt::IdxBtUniformCudaHipBuiltIn<TDim, TIdx>
        , public AtomicHierarchy<
              AtomicUniformCudaHipBuiltIn, // grid atomics
              AtomicUniformCudaHipBuiltIn, // block atomics
              AtomicUniformCudaHipBuiltIn> // thread atomics
        , public math::MathUniformCudaHipBuiltIn
        , public BlockSharedMemDynUniformCudaHipBuiltIn
        , public BlockSharedMemStUniformCudaHipBuiltIn
        , public BlockSyncUniformCudaHipBuiltIn
        , public GridSyncCudaHipBuiltIn
        , public IntrinsicUniformCudaHipBuiltIn
        , public MemFenceUniformCudaHipBuiltIn
#    ifdef AV2P_DISABLE_VENDOR_RNG
        , public rand::RandDefault
#    else
        , public rand::RandUniformCudaHipRand<TApi>
#    endif
        , public warp::WarpUniformCudaHipBuiltIn
        , public interface::Implements<InterfaceAcc, AccGpuUniformCudaHipRt<TApi, TDim, TIdx>>
    {
        static_assert(
            sizeof(TIdx) >= sizeof(int),
            "Index type is not supported, consider using int or a larger type.");

    public:
        AccGpuUniformCudaHipRt(AccGpuUniformCudaHipRt const&) = delete;
        AccGpuUniformCudaHipRt(AccGpuUniformCudaHipRt&&) = delete;
        auto operator=(AccGpuUniformCudaHipRt const&) -> AccGpuUniformCudaHipRt& = delete;
        auto operator=(AccGpuUniformCudaHipRt&&) -> AccGpuUniformCudaHipRt& = delete;

        AV2P_FN_HOST_ACC AccGpuUniformCudaHipRt(Vec<TDim, TIdx> const& threadElemExtent)
            : WorkDivUniformCudaHipBuiltIn<TDim, TIdx>(threadElemExtent)
        {
        }
    };

    namespace trait
    {
        //! The GPU CUDA accelerator accelerator type trait specialization.
        template<typename TApi, typename TDim, typename TIdx>
        struct AccType<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>>
        {
            using type = AccGpuUniformCudaHipRt<TApi, TDim, TIdx>;
        };

        //! The GPU CUDA single thread accelerator type trait specialization.
        template<typename TApi, typename TDim, typename TIdx>
        struct IsSingleThreadAcc<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>> : std::false_type
        {
        };

        //! The GPU CUDA multi thread accelerator type trait specialization.
        template<typename TApi, typename TDim, typename TIdx>
        struct IsMultiThreadAcc<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>> : std::true_type
        {
        };

        //! The GPU CUDA accelerator device properties get trait specialization.
        template<typename TApi, typename TDim, typename TIdx>
        struct GetAccDevProps<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>>
        {
            AV2P_FN_HOST static auto getAccDevProps(DevUniformCudaHipRt<TApi> const& dev) -> AccDevProps<TDim, TIdx>
            {
#    ifdef AV2P_ACC_GPU_CUDA_ENABLED
                // Reading only the necessary attributes with cudaDeviceGetAttribute is faster than reading all with
                // cuda https://devblogs.nvidia.com/cuda-pro-tip-the-fast-way-to-query-device-properties/
                int multiProcessorCount = {};
                AV2P_UNIFORM_CUDA_HIP_RT_CHECK(TApi::deviceGetAttribute(
                    &multiProcessorCount,
                    TApi::deviceAttributeMultiprocessorCount,
                    dev.getNativeHandle()));

                int maxGridSize[3] = {};
                AV2P_UNIFORM_CUDA_HIP_RT_CHECK(TApi::deviceGetAttribute(
                    &maxGridSize[0],
                    TApi::deviceAttributeMaxGridDimX,
                    dev.getNativeHandle()));
                AV2P_UNIFORM_CUDA_HIP_RT_CHECK(TApi::deviceGetAttribute(
                    &maxGridSize[1],
                    TApi::deviceAttributeMaxGridDimY,
                    dev.getNativeHandle()));
                AV2P_UNIFORM_CUDA_HIP_RT_CHECK(TApi::deviceGetAttribute(
                    &maxGridSize[2],
                    TApi::deviceAttributeMaxGridDimZ,
                    dev.getNativeHandle()));

                int maxBlockDim[3] = {};
                AV2P_UNIFORM_CUDA_HIP_RT_CHECK(TApi::deviceGetAttribute(
                    &maxBlockDim[0],
                    TApi::deviceAttributeMaxBlockDimX,
                    dev.getNativeHandle()));
                AV2P_UNIFORM_CUDA_HIP_RT_CHECK(TApi::deviceGetAttribute(
                    &maxBlockDim[1],
                    TApi::deviceAttributeMaxBlockDimY,
                    dev.getNativeHandle()));
                AV2P_UNIFORM_CUDA_HIP_RT_CHECK(TApi::deviceGetAttribute(
                    &maxBlockDim[2],
                    TApi::deviceAttributeMaxBlockDimZ,
                    dev.getNativeHandle()));

                int maxThreadsPerBlock = {};
                AV2P_UNIFORM_CUDA_HIP_RT_CHECK(TApi::deviceGetAttribute(
                    &maxThreadsPerBlock,
                    TApi::deviceAttributeMaxThreadsPerBlock,
                    dev.getNativeHandle()));

                int sharedMemSizeBytes = {};
                AV2P_UNIFORM_CUDA_HIP_RT_CHECK(TApi::deviceGetAttribute(
                    &sharedMemSizeBytes,
                    TApi::deviceAttributeMaxSharedMemoryPerBlock,
                    dev.getNativeHandle()));

                int cooperativeLaunch = {};
                AV2P_UNIFORM_CUDA_HIP_RT_CHECK(TApi::deviceGetAttribute(
                    &cooperativeLaunch,
                    TApi::deviceAttributeCooperativeLaunch,
                    dev.getNativeHandle()));

                return {// m_multiProcessorCount
                        aV2P::core::clipCast<TIdx>(multiProcessorCount),
                        // m_gridBlockExtentMax
                        getExtentVecEnd<TDim>(Vec<DimInt<3u>, TIdx>(
                            aV2P::core::clipCast<TIdx>(maxGridSize[2u]),
                            aV2P::core::clipCast<TIdx>(maxGridSize[1u]),
                            aV2P::core::clipCast<TIdx>(maxGridSize[0u]))),
                        // m_gridBlockCountMax
                        std::numeric_limits<TIdx>::max(),
                        // m_blockThreadExtentMax
                        getExtentVecEnd<TDim>(Vec<DimInt<3u>, TIdx>(
                            aV2P::core::clipCast<TIdx>(maxBlockDim[2u]),
                            aV2P::core::clipCast<TIdx>(maxBlockDim[1u]),
                            aV2P::core::clipCast<TIdx>(maxBlockDim[0u]))),
                        // m_blockThreadCountMax
                        aV2P::core::clipCast<TIdx>(maxThreadsPerBlock),
                        // m_threadElemExtentMax
                        Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
                        // m_threadElemCountMax
                        std::numeric_limits<TIdx>::max(),
                        // m_sharedMemSizeBytes
                        static_cast<size_t>(sharedMemSizeBytes),
                        // m_globalMemSizeBytes
                        getMemBytes(dev),
                        // m_cooperativeLaunch
                        static_cast<bool>(cooperativeLaunch)};

#    else
                typename TApi::DeviceProp_t properties;
                AV2P_UNIFORM_CUDA_HIP_RT_CHECK(TApi::getDeviceProperties(&properties, dev.getNativeHandle()));

                return {// m_multiProcessorCount
                        aV2P::core::clipCast<TIdx>(properties.multiProcessorCount),
                        // m_gridBlockExtentMax
                        getExtentVecEnd<TDim>(Vec<DimInt<3u>, TIdx>(
                            aV2P::core::clipCast<TIdx>(properties.maxGridSize[2u]),
                            aV2P::core::clipCast<TIdx>(properties.maxGridSize[1u]),
                            aV2P::core::clipCast<TIdx>(properties.maxGridSize[0u]))),
                        // m_gridBlockCountMax
                        std::numeric_limits<TIdx>::max(),
                        // m_blockThreadExtentMax
                        getExtentVecEnd<TDim>(Vec<DimInt<3u>, TIdx>(
                            aV2P::core::clipCast<TIdx>(properties.maxThreadsDim[2u]),
                            aV2P::core::clipCast<TIdx>(properties.maxThreadsDim[1u]),
                            aV2P::core::clipCast<TIdx>(properties.maxThreadsDim[0u]))),
                        // m_blockThreadCountMax
                        aV2P::core::clipCast<TIdx>(properties.maxThreadsPerBlock),
                        // m_threadElemExtentMax
                        Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
                        // m_threadElemCountMax
                        std::numeric_limits<TIdx>::max(),
                        // m_sharedMemSizeBytes
                        static_cast<size_t>(properties.sharedMemPerBlock),
                        // m_globalMemSizeBytes
                        getMemBytes(dev),
                        // m_cooperativeLaunch
                        static_cast<bool>(properties.cooperativeLaunch)};
#    endif
            }
        };

        //! The GPU CUDA accelerator name trait specialization.
        template<typename TApi, typename TDim, typename TIdx>
        struct GetAccName<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>>
        {
            AV2P_FN_HOST static auto getAccName() -> std::string
            {
#    if AV2P_COMP_CLANG
#        pragma clang diagnostic push
#        pragma clang diagnostic ignored "-Wexit-time-destructors"
#    endif
                using namespace std::literals;
                static std::string const accName =
#    ifdef __cpp_lib_format
                    std::format("AccGpu{}Rt<{},{}>", TApi::name, TDim::value, core::demangled<TIdx>);
#    else
                    "AccGpu"s + TApi::name + "Rt<"s + std::to_string(TDim::value) + ","s
                    + std::string(core::demangled<TIdx>) + ">"s;
#    endif
                return accName;
#    if AV2P_COMP_CLANG
#        pragma clang diagnostic pop
#    endif
            }
        };

        //! The GPU CUDA accelerator device type trait specialization.
        template<typename TApi, typename TDim, typename TIdx>
        struct DevType<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>>
        {
            using type = DevUniformCudaHipRt<TApi>;
        };

        //! The GPU CUDA accelerator dimension getter trait specialization.
        template<typename TApi, typename TDim, typename TIdx>
        struct DimType<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>>
        {
            using type = TDim;
        };
    } // namespace trait

    namespace trait
    {
        //! The GPU CUDA accelerator execution task type trait specialization.
        template<
            typename TApi,
            typename TDim,
            typename TIdx,
            typename TWorkDiv,
            typename TKernelFnObj,
            typename... TArgs>
        struct CreateTaskKernel<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>, TWorkDiv, TKernelFnObj, TArgs...>
        {
            AV2P_FN_HOST static auto createTaskKernel(
                TWorkDiv const& workDiv,
                TKernelFnObj const& kernelFnObj,
                TArgs&&... args)
            {
                return TaskKernelGpuUniformCudaHipRt<
                    TApi,
                    AccGpuUniformCudaHipRt<TApi, TDim, TIdx>,
                    TDim,
                    TIdx,
                    TKernelFnObj,
                    false,
                    TArgs...>(workDiv, kernelFnObj, std::forward<TArgs>(args)...);
            }
        };

        //! The GPU CUDA accelerator execution cooperative task type trait specialization.
        template<
            typename TApi,
            typename TDim,
            typename TIdx,
            typename TWorkDiv,
            typename TKernelFnObj,
            typename... TArgs>
        struct CreateTaskCooperativeKernel<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>, TWorkDiv, TKernelFnObj, TArgs...>
        {
            AV2P_FN_HOST static auto createTaskCooperativeKernel(
                TWorkDiv const& workDiv,
                TKernelFnObj const& kernelFnObj,
                TArgs&&... args)
            {
                return TaskKernelGpuUniformCudaHipRt<
                    TApi,
                    AccGpuUniformCudaHipRt<TApi, TDim, TIdx>,
                    TDim,
                    TIdx,
                    TKernelFnObj,
                    true,
                    TArgs...>(workDiv, kernelFnObj, std::forward<TArgs>(args)...);
            }
        };

        //! The CPU CUDA execution task platform type trait specialization.
        template<typename TApi, typename TDim, typename TIdx>
        struct PlatformType<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>>
        {
            using type = PlatformUniformCudaHipRt<TApi>;
        };

        //! The GPU CUDA accelerator idx type trait specialization.
        template<typename TApi, typename TDim, typename TIdx>
        struct IdxType<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>>
        {
            using type = TIdx;
        };
    } // namespace trait
} // namespace aV2P

#endif

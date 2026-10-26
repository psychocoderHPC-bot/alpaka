/* Copyright 2025 Benjamin Worpitz, Matthias Werner, René Widera, Bernhard Manfred Gruber, Jan Stephan, Andrea Bocci,
 * Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/alpaka.hpp"

#include <type_traits>

namespace aV2P::test
{
    namespace trait
    {
        //! The default queue type trait for devices.
        template<typename TDev, typename TSfinae = void>
        struct DefaultQueueType;

        //! The default queue type trait specialization for the CPU device.
        template<>
        struct DefaultQueueType<DevCpu>
        {
#if(AV2P_DEBUG >= AV2P_DEBUG_FULL)
            using type = QueueCpuBlocking;
#else
            using type = QueueCpuNonBlocking;
#endif
        };

#ifdef AV2P_ACC_GPU_CUDA_ENABLED
        //! The default queue type trait specialization for the CUDA device.
        template<>
        struct DefaultQueueType<DevCudaRt>
        {
#    if(AV2P_DEBUG >= AV2P_DEBUG_FULL)
            using type = QueueCudaRtBlocking;
#    else
            using type = QueueCudaRtNonBlocking;
#    endif
        };
#endif

#ifdef AV2P_ACC_GPU_HIP_ENABLED
        //! The default queue type trait specialization for the HIP device.
        template<>
        struct DefaultQueueType<DevHipRt>
        {
#    if(AV2P_DEBUG >= AV2P_DEBUG_FULL)
            using type = QueueHipRtBlocking;
#    else
            using type = QueueHipRtNonBlocking;
#    endif
        };
#endif

#ifdef AV2P_ACC_SYCL_ENABLED
        //! The default queue type trait specialization for the SYCL CPU device.
        template<>
        struct DefaultQueueType<DevCpuSycl>
        {
#    if(AV2P_DEBUG >= AV2P_DEBUG_FULL)
            using type = QueueCpuSyclBlocking;
#    else
            using type = QueueCpuSyclNonBlocking;
#    endif
        };

        //! The default queue type trait specialization for the SYCL Intel GPU device.
        template<>
        struct DefaultQueueType<DevGpuSyclIntel>
        {
#    if(AV2P_DEBUG >= AV2P_DEBUG_FULL)
            using type = QueueGpuSyclIntelBlocking;
#    else
            using type = QueueGpuSyclIntelNonBlocking;
#    endif
        };
#endif

        //! The blocking queue trait.
        template<typename TQueue, typename TSfinae = void>
        struct IsBlockingQueue;

        //! The blocking queue trait specialization for a blocking CPU queue.
        template<>
        struct IsBlockingQueue<QueueCpuBlocking>
        {
            static constexpr bool value = true;
        };

        //! The blocking queue trait specialization for a non-blocking CPU queue.
        template<>
        struct IsBlockingQueue<QueueCpuNonBlocking>
        {
            static constexpr bool value = false;
        };

        //! The blocking queue trait specialization for the aV2P queue adapter.
        template<typename TDev, alpaka::concepts::QueueKind TQueueKind>
        struct IsBlockingQueue<QueueAlpaka3<TDev, TQueueKind>>
        {
            static constexpr bool value = std::is_same_v<TQueueKind, alpaka::queueKind::Blocking>;
        };
    } // namespace trait

    //! The queue type that should be used for the given device.
    template<typename TDev>
    using DefaultQueue = typename trait::DefaultQueueType<TDev>::type;

    //! The queue type that should be used for the given accelerator.
    template<typename TQueue>
    using IsBlockingQueue = trait::IsBlockingQueue<TQueue>;

    //! A std::tuple holding tuples of devices and corresponding queue types.
    using TestQueues = std::tuple<
        std::tuple<DevCpu, QueueCpuBlocking>,
        std::tuple<DevCpu, QueueCpuNonBlocking>
#ifdef AV2P_ACC_GPU_CUDA_ENABLED
        ,
        std::tuple<DevCudaRt, QueueCudaRtBlocking>,
        std::tuple<DevCudaRt, QueueCudaRtNonBlocking>
#endif
#ifdef AV2P_ACC_GPU_HIP_ENABLED
        ,
        std::tuple<DevHipRt, QueueHipRtBlocking>,
        std::tuple<DevHipRt, QueueHipRtNonBlocking>
#endif
#ifdef AV2P_ACC_SYCL_ENABLED
#    ifdef AV2P_SYCL_ONEAPI_CPU
        ,
        std::tuple<aV2P::DevCpuSycl, aV2P::QueueCpuSyclBlocking>,
        std::tuple<aV2P::DevCpuSycl, aV2P::QueueCpuSyclNonBlocking>
#    endif
#    ifdef AV2P_SYCL_ONEAPI_FPGA
        ,
        std::tuple<aV2P::DevFpgaSyclIntel, aV2P::QueueFpgaSyclIntelBlocking>,
        std::tuple<aV2P::DevFpgaSyclIntel, aV2P::QueueFpgaSyclIntelNonBlocking>
#    endif
#    ifdef AV2P_SYCL_ONEAPI_GPU
        ,
        std::tuple<aV2P::DevGpuSyclIntel, aV2P::QueueGpuSyclIntelBlocking>,
        std::tuple<aV2P::DevGpuSyclIntel, aV2P::QueueGpuSyclIntelNonBlocking>
#    endif
#    ifdef AV2P_SYCL_ONEAPI_GPU_NVIDIA
        ,
        std::tuple<aV2P::DevGpuSyclNvidia, aV2P::QueueGpuSyclNvidiaBlocking>,
        std::tuple<aV2P::DevGpuSyclNvidia, aV2P::QueueGpuSyclNvidiaNonBlocking>
#    endif
#    ifdef AV2P_SYCL_ONEAPI_GPU_AMD
        ,
        std::tuple<aV2P::DevGpuSyclAmd, aV2P::QueueGpuSyclAmdBlocking>,
        std::tuple<aV2P::DevGpuSyclAmd, aV2P::QueueGpuSyclAmdNonBlocking>
#    endif
#endif
        >;
} // namespace aV2P::test

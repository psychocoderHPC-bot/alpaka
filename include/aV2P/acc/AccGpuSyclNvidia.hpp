/* Copyright 2025 Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/acc/AccGenericSycl.hpp"
#include "aV2P/acc/Tag.hpp"
#include "aV2P/core/Sycl.hpp"

#if defined(AV2P_ACC_SYCL_ENABLED) && defined(AV2P_SYCL_ONEAPI_GPU_NVIDIA)

// Set by CMake for Debug builds with oneAPI 2025.0, see cmake/alpakaCommon.cmake
#    ifdef AV2P_SYCL_ONEAPI_GPU_NVIDIA_DEBUG_UNSUPPORTED
#        error                                                                                                        \
            "The SYCL backend for NVIDIA GPUs does not support Debug builds with oneAPI 2025.0, because ptxas fails on the device debug information. Use a different build type or a different version of oneAPI."
#    endif

namespace aV2P
{
    //! The Nvidia GPU SYCL accelerator.
    //!
    //! This accelerator allows parallel kernel execution on a oneAPI-capable Nvidia GPU target device.
    template<typename TDim, typename TIdx>
    using AccGpuSyclNvidia = AccGenericSycl<TagGpuSyclNvidia, TDim, TIdx>;

    namespace trait
    {
        template<typename TDim, typename TIdx>
        struct AccToTag<aV2P::AccGpuSyclNvidia<TDim, TIdx>>
        {
            using type = aV2P::TagGpuSyclNvidia;
        };

        template<typename TDim, typename TIdx>
        struct TagToAcc<aV2P::TagGpuSyclNvidia, TDim, TIdx>
        {
            using type = aV2P::AccGpuSyclNvidia<TDim, TIdx>;
        };
    } // namespace trait

} // namespace aV2P

#endif

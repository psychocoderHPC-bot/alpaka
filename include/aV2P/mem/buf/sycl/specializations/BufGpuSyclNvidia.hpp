/* Copyright 2025 Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/acc/Tag.hpp"
#include "aV2P/mem/buf/sycl/BufGenericSycl.hpp"

#if defined(AV2P_ACC_SYCL_ENABLED) && defined(AV2P_SYCL_ONEAPI_GPU_NVIDIA)

namespace aV2P
{
    template<typename TElem, typename TDim, typename TIdx>
    using ConstBufGpuSyclNvidia = ConstBufGenericSycl<TElem, TDim, TIdx, TagGpuSyclNvidia>;

    template<typename TElem, typename TDim, typename TIdx>
    using BufGpuSyclNvidia = BufGenericSycl<TElem, TDim, TIdx, TagGpuSyclNvidia>;
} // namespace aV2P

#endif

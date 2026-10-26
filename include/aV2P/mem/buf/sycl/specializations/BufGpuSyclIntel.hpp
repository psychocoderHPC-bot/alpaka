/* Copyright 2024 Jan Stephan, Luca Ferragina, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/acc/Tag.hpp"
#include "aV2P/mem/buf/sycl/BufGenericSycl.hpp"

#if defined(AV2P_ACC_SYCL_ENABLED) && defined(AV2P_SYCL_ONEAPI_GPU)

namespace aV2P
{
    template<typename TElem, typename TDim, typename TIdx>
    using ConstBufGpuSyclIntel = ConstBufGenericSycl<TElem, TDim, TIdx, TagGpuSyclIntel>;

    template<typename TElem, typename TDim, typename TIdx>
    using BufGpuSyclIntel = BufGenericSycl<TElem, TDim, TIdx, TagGpuSyclIntel>;
} // namespace aV2P

#endif

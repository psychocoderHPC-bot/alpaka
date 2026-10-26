/* Copyright 2024 Jan Stephan, Luca Ferragina, Andrea Bocci, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/acc/Tag.hpp"
#include "aV2P/mem/buf/sycl/BufGenericSycl.hpp"

#if defined(AV2P_ACC_SYCL_ENABLED) && defined(AV2P_SYCL_ONEAPI_CPU)

namespace aV2P
{
    template<typename TElem, typename TDim, typename TIdx>
    using ConstBufCpuSycl = ConstBufGenericSycl<TElem, TDim, TIdx, TagCpuSycl>;

    template<typename TElem, typename TDim, typename TIdx>
    using BufCpuSycl = BufGenericSycl<TElem, TDim, TIdx, TagCpuSycl>;
} // namespace aV2P

#endif

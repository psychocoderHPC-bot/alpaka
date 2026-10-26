/* Copyright 2024 Jan Stephan, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/acc/Tag.hpp"
#include "aV2P/mem/buf/sycl/BufGenericSycl.hpp"

#if defined(AV2P_ACC_SYCL_ENABLED) && defined(AV2P_SYCL_ONEAPI_FPGA)

namespace aV2P
{
    template<typename TElem, typename TDim, typename TIdx>
    using ConstBufFpgaSyclIntel = ConstBufGenericSycl<TElem, TDim, TIdx, TagFpgaSyclIntel>;

    template<typename TElem, typename TDim, typename TIdx>
    using BufFpgaSyclIntel = BufGenericSycl<TElem, TDim, TIdx, TagFpgaSyclIntel>;
} // namespace aV2P

#endif

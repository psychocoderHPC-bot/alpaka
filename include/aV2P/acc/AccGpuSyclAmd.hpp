/* Copyright 2025 Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/acc/AccGenericSycl.hpp"
#include "aV2P/acc/Tag.hpp"
#include "aV2P/core/Sycl.hpp"

#if defined(AV2P_ACC_SYCL_ENABLED) && defined(AV2P_SYCL_ONEAPI_GPU_AMD)

namespace aV2P
{
    //! The Amd GPU SYCL accelerator.
    //!
    //! This accelerator allows parallel kernel execution on a oneAPI-capable Amd GPU target device.
    template<typename TDim, typename TIdx>
    using AccGpuSyclAmd = AccGenericSycl<TagGpuSyclAmd, TDim, TIdx>;

    namespace trait
    {
        template<typename TDim, typename TIdx>
        struct AccToTag<aV2P::AccGpuSyclAmd<TDim, TIdx>>
        {
            using type = aV2P::TagGpuSyclAmd;
        };

        template<typename TDim, typename TIdx>
        struct TagToAcc<aV2P::TagGpuSyclAmd, TDim, TIdx>
        {
            using type = aV2P::AccGpuSyclAmd<TDim, TIdx>;
        };
    } // namespace trait

} // namespace aV2P

#endif

/* Copyright 2024 Jan Stephan, Luca Ferragina, Andrea Bocci, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/acc/AccGenericSycl.hpp"
#include "aV2P/acc/Tag.hpp"
#include "aV2P/core/Sycl.hpp"

#if defined(AV2P_ACC_SYCL_ENABLED) && defined(AV2P_SYCL_ONEAPI_CPU)

namespace aV2P
{
    //! The CPU SYCL accelerator.
    //!
    //! This accelerator allows parallel kernel execution on a oneAPI-capable CPU target device.
    template<typename TDim, typename TIdx>
    using AccCpuSycl = AccGenericSycl<TagCpuSycl, TDim, TIdx>;

    namespace trait
    {
        template<typename TDim, typename TIdx>
        struct AccToTag<aV2P::AccCpuSycl<TDim, TIdx>>
        {
            using type = aV2P::TagCpuSycl;
        };

        template<typename TDim, typename TIdx>
        struct TagToAcc<aV2P::TagCpuSycl, TDim, TIdx>
        {
            using type = aV2P::AccCpuSycl<TDim, TIdx>;
        };
    } // namespace trait

} // namespace aV2P

#endif

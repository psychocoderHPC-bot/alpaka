/* Copyright 2024 Jan Stephan, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/acc/AccGenericSycl.hpp"
#include "aV2P/acc/Tag.hpp"
#include "aV2P/core/Sycl.hpp"

#if defined(AV2P_ACC_SYCL_ENABLED) && defined(AV2P_SYCL_ONEAPI_FPGA)

namespace aV2P
{
    //! The Intel FPGA SYCL accelerator.
    //!
    //! This accelerator allows parallel kernel execution on a oneAPI-capable Intel FPGA target device.
    template<typename TDim, typename TIdx>
    using AccFpgaSyclIntel = AccGenericSycl<TagFpgaSyclIntel, TDim, TIdx>;

    namespace trait
    {
        template<typename TDim, typename TIdx>
        struct AccToTag<aV2P::AccFpgaSyclIntel<TDim, TIdx>>
        {
            using type = aV2P::TagFpgaSyclIntel;
        };

        template<typename TDim, typename TIdx>
        struct TagToAcc<aV2P::TagFpgaSyclIntel, TDim, TIdx>
        {
            using type = aV2P::AccFpgaSyclIntel<TDim, TIdx>;
        };
    } // namespace trait

} // namespace aV2P

#endif

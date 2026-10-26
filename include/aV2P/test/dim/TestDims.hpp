/* Copyright 2023 Benjamin Worpitz, Andrea Bocci, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Config.hpp"
#include "aV2P/dim/DimIntegralConst.hpp"
#include "aV2P/meta/Filter.hpp"
#include "aV2P/meta/NonZero.hpp"

#include <tuple>

namespace aV2P::test
{
    //! A std::tuple holding dimensions.
    using TestDims = std::tuple<
        DimInt<0u>,
        DimInt<1u>,
        DimInt<2u>,
        DimInt<3u>
    // CUDA, HIP and SYCL accelerators do not support 4D buffers and 4D acceleration.
#if !defined(AV2P_ACC_GPU_CUDA_ENABLED) && !defined(AV2P_ACC_GPU_HIP_ENABLED) && !defined(AV2P_ACC_SYCL_ENABLED)
        ,
        DimInt<4u>
#endif
        >;

    //! A std::tuple holding non-zero dimensions.
    //!
    //! NonZeroTestDims = std::tuple<Dim1, Dim2, ... DimN>
    using NonZeroTestDims = meta::Filter<TestDims, meta::NonZero>;

} // namespace aV2P::test

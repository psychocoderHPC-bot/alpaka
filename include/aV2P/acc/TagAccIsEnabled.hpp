/* Copyright 2025 Simeon Ehrig, Simone Balducci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// include all Acc's because of the struct AccIsEnabled
// if an acc is not include, it will be not enabled independent of the compiler flags
#include "aV2P/acc/AccCpuOmp2Blocks.hpp"
#include "aV2P/acc/AccCpuOmp2Threads.hpp"
#include "aV2P/acc/AccCpuSerial.hpp"
#include "aV2P/acc/AccCpuSycl.hpp"
#include "aV2P/acc/AccCpuTbbBlocks.hpp"
#include "aV2P/acc/AccCpuThreads.hpp"
#include "aV2P/acc/AccFpgaSyclIntel.hpp"
#include "aV2P/acc/AccGpuCudaRt.hpp"
#include "aV2P/acc/AccGpuHipRt.hpp"
#include "aV2P/acc/AccGpuUniformCudaHipRt.hpp"
#include "aV2P/acc/Tag.hpp"
#include "aV2P/dim/DimIntegralConst.hpp"
#include "aV2P/meta/Filter.hpp"

#include <type_traits>

namespace aV2P
{
    //! \brief check if the accelerator is enabled for a given tag
    //! \tparam TTag alpaka tag type
    template<concepts::Tag TTag, typename = void>
    struct AccIsEnabled : std::false_type
    {
    };

    template<concepts::Tag TTag>
    struct AccIsEnabled<TTag, std::void_t<TagToAcc<TTag, aV2P::DimInt<1>, int>>> : std::true_type
    {
    };

    //! list of all tags where the related accelerator is enabled
    using EnabledAccTags = aV2P::meta::Filter<AccTags, aV2P::AccIsEnabled>;

    namespace trait
    {

        template<concepts::Tag TTag>
        struct DevType<TTag>
        {
            using type = typename DevType<aV2P::TagToAcc<TTag, aV2P::DimInt<1>, int>>::type;
        };

    } // namespace trait

} // namespace aV2P

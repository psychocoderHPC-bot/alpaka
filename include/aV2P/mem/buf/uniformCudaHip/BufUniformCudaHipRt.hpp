/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#if defined(AV2P_ACC_GPU_CUDA_ENABLED) || defined(AV2P_ACC_GPU_HIP_ENABLED)

#    include "aV2P/dev/Traits.hpp"
#    include "aV2P/mem/buf/Traits.hpp"
#    include "aV2P/mem/buf/uniformCudaHip/BufUniformCudaHipRt.hpp"
#    include "aV2P/mem/buf/uniformCudaHip/BufUniformCudaHipRtImpl.hpp"
#    include "aV2P/mem/buf/uniformCudaHip/ConstBufUniformCudaHipRt.hpp"
#    include "aV2P/mem/view/ViewAccessOps.hpp"
#    include "aV2P/vec/Vec.hpp"

#    include <functional>
#    include <memory>
#    include <type_traits>
#    include <utility>

namespace aV2P
{

    //! The generic memory buffer template implementing muting accessors.
    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    class BufUniformCudaHipRt
        : public internal::ViewAccessorType<DevUniformCudaHipRt<TApi>, BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>
    {
        using TBufImpl = detail::BufUniformCudaHipRtImpl<TApi, TElem, TDim, TIdx>;

    public:
        template<typename TExtent, typename Deleter>
        AV2P_FN_HOST BufUniformCudaHipRt(
            DevUniformCudaHipRt<TApi> const& dev,
            TElem* const pMem,
            Deleter deleter,
            TExtent const& extent,
            std::size_t pitchBytes)
            : m_spBufImpl{std::make_shared<TBufImpl>(dev, pMem, std::move(deleter), extent, pitchBytes)}
        {
        }

    public:
        std::shared_ptr<TBufImpl> m_spBufImpl;
    };

} // namespace aV2P

#    include "aV2P/mem/buf/uniformCudaHip/Copy.hpp"
#    include "aV2P/mem/buf/uniformCudaHip/Fill.hpp"
#    include "aV2P/mem/buf/uniformCudaHip/Set.hpp"
#    include "aV2P/mem/buf/uniformCudaHip/traits/BufUniformCudaHipRtTraits.hpp"
#    include "aV2P/mem/buf/uniformCudaHip/traits/ConstBufUniformCudaHipRtTraits.hpp"

#endif

/* Copyright 2025 Alexander Matthes, Benjamin Worpitz, Matthias Werner, René Widera, Andrea Bocci, Jan Stephan,
 *                Bernhard Manfred Gruber, Antonio Di Pilato, Anton Reinhard
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Assert.hpp"
#include "aV2P/core/Cuda.hpp"
#include "aV2P/core/Hip.hpp"
#include "aV2P/dev/DevUniformCudaHipRt.hpp"
#include "aV2P/dev/Traits.hpp"
#include "aV2P/dim/DimIntegralConst.hpp"
#include "aV2P/mem/buf/Traits.hpp"
#include "aV2P/mem/buf/uniformCudaHip/BufUniformCudaHipRtImpl.hpp"
#include "aV2P/mem/view/ViewAccessOps.hpp"
#include "aV2P/meta/DependentFalseType.hpp"
#include "aV2P/vec/Vec.hpp"

#include <cstddef>
#include <functional>
#include <memory>
#include <type_traits>

#if defined(AV2P_ACC_GPU_CUDA_ENABLED) || defined(AV2P_ACC_GPU_HIP_ENABLED)

namespace aV2P
{
    // Forward declarations.
    struct ApiCudaRt;
    struct ApiHipRt;
    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    class BufUniformCudaHipRt;

    //! The CUDA/HIP memory buffer.
    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    struct ConstBufUniformCudaHipRt
        : internal::ViewAccessorType<DevUniformCudaHipRt<TApi>, ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>
    {
        static_assert(!std::is_const_v<TElem>, "The elem type of the buffer must not be const");
        static_assert(!std::is_const_v<TIdx>, "The idx type of the buffer must not be const!");

        //! Constructor
        template<typename TExtent, typename Deleter>
        AV2P_FN_HOST ConstBufUniformCudaHipRt(
            DevUniformCudaHipRt<TApi> const& dev,
            TElem* const pMem,
            Deleter deleter,
            TExtent const& extent,
            std::size_t pitchBytes)
            : m_spBufImpl{std::make_shared<detail::BufUniformCudaHipRtImpl<TApi, TElem, TDim, TIdx>>(
                dev,
                pMem,
                std::move(deleter),
                extent,
                pitchBytes)}
        {
        }

        AV2P_FN_HOST ConstBufUniformCudaHipRt(BufUniformCudaHipRt<TApi, TElem, TDim, TIdx> const& buf)
            : m_spBufImpl{buf.m_spBufImpl}
        {
        }

        AV2P_FN_HOST ConstBufUniformCudaHipRt(BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>&& buf)
            : m_spBufImpl{std::move(buf.m_spBufImpl)}
        {
        }

    private:
        std::shared_ptr<detail::BufUniformCudaHipRtImpl<TApi, TElem, TDim, TIdx>> m_spBufImpl;

        friend aV2P::trait::GetDev<ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>;
        friend aV2P::trait::GetExtents<ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>;
        friend aV2P::trait::GetPtrNative<ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>;
        friend aV2P::trait::GetPtrDev<ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>, DevUniformCudaHipRt<TApi>>;
        friend aV2P::trait::GetPitchesInBytes<ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>;
    };

} // namespace aV2P

#    include "aV2P/mem/buf/uniformCudaHip/Copy.hpp"
#    include "aV2P/mem/buf/uniformCudaHip/Fill.hpp"
#    include "aV2P/mem/buf/uniformCudaHip/Set.hpp"

#endif

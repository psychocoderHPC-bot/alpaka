/* Copyright 2022 Alexander Matthes, Axel Huebl, Benjamin Worpitz, Andrea Bocci, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/ApiCudaRt.hpp"
#include "aV2P/core/ApiHipRt.hpp"
#include "aV2P/core/Cuda.hpp"
#include "aV2P/core/Hip.hpp"
#include "aV2P/core/Vectorize.hpp"
#include "aV2P/dev/DevCpu.hpp"
#include "aV2P/dev/Traits.hpp"
#include "aV2P/mem/alloc/AllocCpuAligned.hpp"
#include "aV2P/mem/buf/Traits.hpp"
#include "aV2P/mem/buf/cpu/BufCpuImpl.hpp"
#include "aV2P/mem/view/ViewAccessOps.hpp"
#include "aV2P/meta/DependentFalseType.hpp"
#include "aV2P/platform/PlatformCpu.hpp"
#include "aV2P/vec/Vec.hpp"

#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

namespace aV2P
{
    // Predeclaration of BufCpu
    template<typename TElem, typename TDim, typename TIdx>
    class BufCpu;

    //! The CPU memory buffer.
    template<typename TElem, typename TDim, typename TIdx>
    class ConstBufCpu : public internal::ViewAccessorType<DevCpu, ConstBufCpu<TElem, TDim, TIdx>>
    {
    public:
        template<typename TExtent, typename Deleter>
        AV2P_FN_HOST ConstBufCpu(DevCpu const& dev, TElem* pMem, Deleter deleter, TExtent const& extent)
            : m_spBufImpl{
                std::make_shared<detail::BufCpuImpl<TElem, TDim, TIdx>>(dev, pMem, std::move(deleter), extent)}
        {
        }

        AV2P_FN_HOST ConstBufCpu(BufCpu<TElem, TDim, TIdx> const& buf) : m_spBufImpl{buf.m_spBufImpl}
        {
        }

        AV2P_FN_HOST ConstBufCpu(BufCpu<TElem, TDim, TIdx>&& buf) : m_spBufImpl{std::move(buf.m_spBufImpl)}
        {
        }

    private:
        std::shared_ptr<detail::BufCpuImpl<TElem, TDim, TIdx>> m_spBufImpl;

        friend aV2P::trait::GetDev<ConstBufCpu<TElem, TDim, TIdx>>;
        friend aV2P::trait::GetExtents<ConstBufCpu<TElem, TDim, TIdx>>;
        friend aV2P::trait::GetPtrNative<ConstBufCpu<TElem, TDim, TIdx>>;
        friend aV2P::trait::GetPtrDev<ConstBufCpu<TElem, TDim, TIdx>, DevCpu>;
    };

} // namespace aV2P

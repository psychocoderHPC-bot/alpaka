/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/dev/Traits.hpp"
#include "aV2P/mem/buf/Traits.hpp"
#include "aV2P/mem/buf/cpu/BufCpuImpl.hpp"
#include "aV2P/mem/buf/cpu/ConstBufCpu.hpp"
#include "aV2P/mem/view/ViewAccessOps.hpp"
#include "aV2P/vec/Vec.hpp"

#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

namespace aV2P
{
    //! The CPU memory buffer template implementing muting accessors.
    template<typename TElem, typename TDim, typename TIdx>
    class BufCpu : public internal::ViewAccessorType<DevCpu, BufCpu<TElem, TDim, TIdx>>
    {
        using TBufImpl = detail::BufCpuImpl<TElem, TDim, TIdx>;

    public:
        template<typename TExtent, typename Deleter>
        AV2P_FN_HOST BufCpu(DevCpu const& dev, TElem* const pMem, Deleter deleter, TExtent const& extent)
            : m_spBufImpl{std::make_shared<TBufImpl>(dev, pMem, std::move(deleter), extent)}
        {
        }

    public:
        std::shared_ptr<TBufImpl> m_spBufImpl;
    };
} // namespace aV2P

#include "aV2P/mem/buf/cpu/Copy.hpp"
#include "aV2P/mem/buf/cpu/Fill.hpp"
#include "aV2P/mem/buf/cpu/Set.hpp"
#include "aV2P/mem/buf/cpu/traits/BufCpuTraits.hpp"
#include "aV2P/mem/buf/cpu/traits/ConstBufCpuTraits.hpp"

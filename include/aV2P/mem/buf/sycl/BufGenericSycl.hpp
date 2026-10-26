/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/dev/DevCpu.hpp"
#include "aV2P/dev/Traits.hpp"
#include "aV2P/mem/buf/Traits.hpp"
#include "aV2P/mem/buf/cpu/BufCpu.hpp"
#include "aV2P/mem/buf/sycl/BufGenericSyclImpl.hpp"
#include "aV2P/mem/buf/sycl/ConstBufGenericSycl.hpp"
#include "aV2P/mem/view/ViewAccessOps.hpp"
#include "aV2P/vec/Vec.hpp"

#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

#ifdef AV2P_ACC_SYCL_ENABLED

namespace aV2P
{
    //! The generic memory buffer template implementing muting accessors.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    class BufGenericSycl
        : public internal::ViewAccessorType<DevGenericSycl<TTag>, BufGenericSycl<TElem, TDim, TIdx, TTag>>
    {
        using TBufImpl = detail::BufGenericSyclImpl<TElem, TDim, TIdx, TTag>;

    public:
        template<typename TExtent, typename Deleter>
        AV2P_FN_HOST BufGenericSycl(
            DevGenericSycl<TTag> const& dev,
            TElem* const pMem,
            Deleter deleter,
            TExtent const& extent)
            : m_spBufImpl{std::make_shared<TBufImpl>(dev, pMem, std::move(deleter), extent)}
        {
        }

    public:
        std::shared_ptr<TBufImpl> m_spBufImpl;
    };
} // namespace aV2P

#    include "aV2P/mem/buf/sycl/Copy.hpp"
#    include "aV2P/mem/buf/sycl/Fill.hpp"
#    include "aV2P/mem/buf/sycl/Set.hpp"
#    include "aV2P/mem/buf/sycl/traits/BufGenericSyclTraits.hpp"
#    include "aV2P/mem/buf/sycl/traits/ConstBufGenericSyclTraits.hpp"

#endif

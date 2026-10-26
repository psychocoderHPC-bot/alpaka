/* Copyright 2024 Jan Stephan, Luca Ferragina, Aurora Perego, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Sycl.hpp"
#include "aV2P/dev/DevGenericSycl.hpp"
#include "aV2P/dev/Traits.hpp"
#include "aV2P/dim/DimIntegralConst.hpp"
#include "aV2P/dim/Traits.hpp"
#include "aV2P/mem/buf/Traits.hpp"
#include "aV2P/mem/buf/sycl/BufGenericSyclImpl.hpp"
#include "aV2P/mem/view/ViewAccessOps.hpp"
#include "aV2P/vec/Vec.hpp"

#include <memory>
#include <type_traits>

#ifdef AV2P_ACC_SYCL_ENABLED

#    include <sycl/sycl.hpp>

namespace aV2P
{
    // Predeclaration
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    class BufGenericSycl;

    //! The SYCL memory buffer.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    class ConstBufGenericSycl
        : public internal::ViewAccessorType<DevGenericSycl<TTag>, ConstBufGenericSycl<TElem, TDim, TIdx, TTag>>
    {
    public:
        //! Constructor
        template<typename TExtent, typename Deleter>
        ConstBufGenericSycl(DevGenericSycl<TTag> const& dev, TElem* pMem, Deleter deleter, TExtent const& extent)
            : m_spBufImpl{std::make_shared<detail::BufGenericSyclImpl<TElem, TDim, TIdx, TTag>>(
                dev,
                pMem,
                std::move(deleter),
                extent)}
        {
        }

        //! Constructor for a ConstBuf from a BufGenericSycl
        AV2P_FN_HOST ConstBufGenericSycl(BufGenericSycl<TElem, TDim, TIdx, TTag> const& buf)
            : m_spBufImpl{buf.m_spBufImpl}
        {
        }

        //! Constructor for a ConstBuf from a BufGenericSycl
        AV2P_FN_HOST ConstBufGenericSycl(BufGenericSycl<TElem, TDim, TIdx, TTag>&& buf)
            : m_spBufImpl{std::move(buf.m_spBufImpl)}
        {
        }

    private:
        std::shared_ptr<detail::BufGenericSyclImpl<TElem, TDim, TIdx, TTag>> m_spBufImpl;

        friend aV2P::trait::GetDev<ConstBufGenericSycl<TElem, TDim, TIdx, TTag>>;
        friend aV2P::trait::GetExtents<ConstBufGenericSycl<TElem, TDim, TIdx, TTag>>;
        friend aV2P::trait::GetPtrNative<ConstBufGenericSycl<TElem, TDim, TIdx, TTag>>;
        friend aV2P::trait::GetPtrDev<ConstBufGenericSycl<TElem, TDim, TIdx, TTag>, DevGenericSycl<TTag>>;
    };
} // namespace aV2P

#endif

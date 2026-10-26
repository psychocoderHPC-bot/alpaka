/* Copyright 2023 Jan Stephan, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Interface.hpp"
#include "aV2P/core/Positioning.hpp"
#include "aV2P/core/Sycl.hpp"
#include "aV2P/idx/Traits.hpp"
#include "aV2P/vec/Vec.hpp"

#ifdef AV2P_ACC_SYCL_ENABLED

#    include <sycl/sycl.hpp>

namespace aV2P::bt
{
    //! The SYCL accelerator ND index provider.
    template<typename TDim, typename TIdx>
    class IdxBtGenericSycl : public interface::Implements<ConceptIdxBt, IdxBtGenericSycl<TDim, TIdx>>
    {
    public:
        using IdxBtBase = IdxBtGenericSycl;

        explicit IdxBtGenericSycl(sycl::nd_item<TDim::value> work_item) : m_item_bt{work_item}
        {
        }

        sycl::nd_item<TDim::value> m_item_bt;
    };
} // namespace aV2P::bt

namespace aV2P::trait
{
    //! The SYCL accelerator index dimension get trait specialization.
    template<typename TDim, typename TIdx>
    struct DimType<bt::IdxBtGenericSycl<TDim, TIdx>>
    {
        using type = TDim;
    };

    //! The SYCL accelerator block thread index get trait specialization.
    template<typename TDim, typename TIdx>
    struct GetIdx<bt::IdxBtGenericSycl<TDim, TIdx>, origin::Block, unit::Threads>
    {
        //! \return The index of the current thread in the block.
        template<typename TWorkDiv>
        static auto getIdx(bt::IdxBtGenericSycl<TDim, TIdx> const& idx, TWorkDiv const&) -> Vec<TDim, TIdx>
        {
            if constexpr(TDim::value == 1)
                return Vec<TDim, TIdx>{static_cast<TIdx>(idx.m_item_bt.get_local_id(0))};
            else if constexpr(TDim::value == 2)
            {
                return Vec<TDim, TIdx>{
                    static_cast<TIdx>(idx.m_item_bt.get_local_id(0)),
                    static_cast<TIdx>(idx.m_item_bt.get_local_id(1))};
            }
            else
            {
                return Vec<TDim, TIdx>{
                    static_cast<TIdx>(idx.m_item_bt.get_local_id(0)),
                    static_cast<TIdx>(idx.m_item_bt.get_local_id(1)),
                    static_cast<TIdx>(idx.m_item_bt.get_local_id(2))};
            }
        }
    };

    //! The SYCL accelerator block thread index idx type trait specialization.
    template<typename TDim, typename TIdx>
    struct IdxType<bt::IdxBtGenericSycl<TDim, TIdx>>
    {
        using type = TIdx;
    };
} // namespace aV2P::trait

#endif

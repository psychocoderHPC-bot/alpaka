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

namespace aV2P::gb
{
    //! The SYCL accelerator ND index provider.
    template<typename TDim, typename TIdx>
    class IdxGbGenericSycl : public interface::Implements<ConceptIdxGb, IdxGbGenericSycl<TDim, TIdx>>
    {
    public:
        using IdxGbBase = IdxGbGenericSycl;

        explicit IdxGbGenericSycl(sycl::nd_item<TDim::value> work_item) : m_item_gb{work_item}
        {
        }

        sycl::nd_item<TDim::value> m_item_gb;
    };
} // namespace aV2P::gb

namespace aV2P::trait
{
    //! The SYCL accelerator index dimension get trait specialization.
    template<typename TDim, typename TIdx>
    struct DimType<gb::IdxGbGenericSycl<TDim, TIdx>>
    {
        using type = TDim;
    };

    //! The SYCL accelerator grid block index get trait specialization.
    template<typename TDim, typename TIdx>
    struct GetIdx<gb::IdxGbGenericSycl<TDim, TIdx>, origin::Grid, unit::Blocks>
    {
        //! \return The index of the current block in the grid.
        template<typename TWorkDiv>
        static auto getIdx(gb::IdxGbGenericSycl<TDim, TIdx> const& idx, TWorkDiv const&)
        {
            if constexpr(TDim::value == 1)
                return Vec<TDim, TIdx>(static_cast<TIdx>(idx.m_item_gb.get_group(0)));
            else if constexpr(TDim::value == 2)
            {
                return Vec<TDim, TIdx>(
                    static_cast<TIdx>(idx.m_item_gb.get_group(0)),
                    static_cast<TIdx>(idx.m_item_gb.get_group(1)));
            }
            else
            {
                return Vec<TDim, TIdx>(
                    static_cast<TIdx>(idx.m_item_gb.get_group(0)),
                    static_cast<TIdx>(idx.m_item_gb.get_group(1)),
                    static_cast<TIdx>(idx.m_item_gb.get_group(2)));
            }
        }
    };

    //! The SYCL accelerator grid block index idx type trait specialization.
    template<typename TDim, typename TIdx>
    struct IdxType<gb::IdxGbGenericSycl<TDim, TIdx>>
    {
        using type = TIdx;
    };
} // namespace aV2P::trait

#endif

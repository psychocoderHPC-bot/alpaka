/* Copyright 2022 Jeffrey Kelling, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/idx/Accessors.hpp"
#include "aV2P/idx/MapIdx.hpp"
#include "aV2P/idx/Traits.hpp"
#include "aV2P/mem/buf/Traits.hpp"
#include "aV2P/meta/Fold.hpp"

namespace aV2P
{
    //! any device ND memory set kernel.
    class MemSetKernel
    {
    public:
        //! The kernel entry point.
        //!
        //! All but the last element of threadElemExtent must be one.
        //!
        //! \tparam TAcc The accelerator environment to be executed on.
        //! \tparam TExtent extent type.
        //! \param acc The accelerator to be executed on.
        //! \param val value to set.
        //! \param dst target mem ptr.
        //! \param extent area to set.
        AV2P_NO_HOST_ACC_WARNING
        template<typename TAcc, typename TExtent, typename TPitch>
        AV2P_FN_ACC auto operator()(
            TAcc const& acc,
            std::uint8_t const val,
            std::uint8_t* dst,
            TExtent extent,
            TPitch pitch) const -> void
        {
            using Idx = typename aV2P::trait::IdxType<TExtent>::type;
            auto const gridThreadIdx(aV2P::getIdx<aV2P::Grid, aV2P::Threads>(acc));
            auto const threadElemExtent(aV2P::getWorkDiv<aV2P::Thread, aV2P::Elems>(acc));
            auto const idxThreadFirstElem = getIdxThreadFirstElem(acc, gridThreadIdx, threadElemExtent);
            auto idx = mapIdxPitchBytes<1u, Dim<TAcc>::value>(idxThreadFirstElem, pitch)[0];
            constexpr auto lastDim = Dim<TAcc>::value - 1;
            auto const lastIdx = idx
                                 + std::min(
                                     threadElemExtent[lastDim],
                                     static_cast<Idx>(extent[lastDim] - idxThreadFirstElem[lastDim]));

            if((idxThreadFirstElem < extent).foldrAll(std::logical_and<bool>()))
            {
                for(; idx < lastIdx; ++idx)
                {
                    *(dst + idx) = val;
                }
            }
        }
    };
} // namespace aV2P

/* Copyright 2022 Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Common.hpp"
#include "aV2P/core/Interface.hpp"
#include "aV2P/core/Positioning.hpp"
#include "aV2P/idx/Traits.hpp"
#include "aV2P/vec/Vec.hpp"

#include <type_traits>
#include <utility>

namespace aV2P
{
    struct ConceptWorkDiv
    {
    };

    //! The work division trait.
    namespace trait
    {
        //! The work div trait.
        template<typename TWorkDiv, typename TOrigin, typename TUnit, typename TSfinae = void>
        struct GetWorkDiv;
    } // namespace trait

    //! Get the extent requested.
    AV2P_NO_HOST_ACC_WARNING
    template<typename TOrigin, typename TUnit, typename TWorkDiv>
    AV2P_FN_HOST_ACC auto getWorkDiv(TWorkDiv const& workDiv) -> Vec<Dim<TWorkDiv>, Idx<TWorkDiv>>
    {
        using ImplementationBase = interface::ImplementationBase<ConceptWorkDiv, TWorkDiv>;
        return trait::GetWorkDiv<ImplementationBase, TOrigin, TUnit>::getWorkDiv(workDiv);
    }

    namespace trait
    {
        //! The work div grid thread extent trait specialization.
        template<typename TWorkDiv>
        struct GetWorkDiv<TWorkDiv, origin::Grid, unit::Threads>
        {
            AV2P_NO_HOST_ACC_WARNING
            AV2P_FN_HOST_ACC static auto getWorkDiv(TWorkDiv const& workDiv)
            {
                return aV2P::getWorkDiv<origin::Grid, unit::Blocks>(workDiv)
                       * aV2P::getWorkDiv<origin::Block, unit::Threads>(workDiv);
            }
        };

        //! The work div grid element extent trait specialization.
        template<typename TWorkDiv>
        struct GetWorkDiv<TWorkDiv, origin::Grid, unit::Elems>
        {
            AV2P_NO_HOST_ACC_WARNING
            AV2P_FN_HOST_ACC static auto getWorkDiv(TWorkDiv const& workDiv)
            {
                return aV2P::getWorkDiv<origin::Grid, unit::Threads>(workDiv)
                       * aV2P::getWorkDiv<origin::Thread, unit::Elems>(workDiv);
            }
        };

        //! The work div block element extent trait specialization.
        template<typename TWorkDiv>
        struct GetWorkDiv<TWorkDiv, origin::Block, unit::Elems>
        {
            AV2P_NO_HOST_ACC_WARNING
            AV2P_FN_HOST_ACC static auto getWorkDiv(TWorkDiv const& workDiv)
            {
                return aV2P::getWorkDiv<origin::Block, unit::Threads>(workDiv)
                       * aV2P::getWorkDiv<origin::Thread, unit::Elems>(workDiv);
            }
        };
    } // namespace trait
} // namespace aV2P

/* Copyright 2022 Benjamin Worpitz, Matthias Werner, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/alpaka.hpp"

#include <cstddef>

namespace aV2P::test
{
#if AV2P_COMP_CLANG >= AV2P_VERSION_NUMBER(21, 1, 0)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wnrvo"
#endif
    template<typename TDim, typename TVal>
    inline constexpr auto extentBuf = []
    {
        Vec<TDim, TVal> v;
        if constexpr(TDim::value > 0)
            for(TVal i = 0; i < TVal{TDim::value}; i++)
                v[i] = 11 - i;
        return v;
    }();

    template<typename TDim, typename TVal>
    inline constexpr auto extentSubView = []
    {
        Vec<TDim, TVal> v;
        if constexpr(TDim::value > 0)
            for(TVal i = 0; i < TVal{TDim::value}; i++)
                v[i] = 8 - i * 2;
        return v;
    }();

    template<typename TDim, typename TVal>
    inline constexpr auto offset = []
    {
        Vec<TDim, TVal> v;
        if constexpr(TDim::value > 0)
            for(TVal i = 0; i < TVal{TDim::value}; i++)
                v[i] = 2 + i;
        return v;
    }();
#if AV2P_COMP_CLANG >= AV2P_VERSION_NUMBER(21, 1, 0)
#    pragma clang diagnostic pop
#endif
} // namespace aV2P::test

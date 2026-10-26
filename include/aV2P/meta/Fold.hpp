/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Common.hpp"

namespace aV2P::meta
{
    AV2P_NO_HOST_ACC_WARNING
    template<typename TFnObj, typename T>
    AV2P_FN_HOST_ACC constexpr auto foldr(TFnObj const& /* f */, T const& t) -> T
    {
        return t;
    }

    AV2P_NO_HOST_ACC_WARNING
    template<typename TFnObj, typename T0, typename T1, typename... Ts>
    AV2P_FN_HOST_ACC constexpr auto foldr(TFnObj const& f, T0 const& t0, T1 const& t1, Ts const&... ts)
    {
        return f(t0, foldr(f, t1, ts...));
    }
} // namespace aV2P::meta

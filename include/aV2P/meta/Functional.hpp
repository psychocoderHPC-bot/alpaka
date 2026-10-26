/* Copyright 2022 Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Common.hpp"

namespace aV2P::meta
{
    template<typename T>
    struct min
    {
        AV2P_NO_HOST_ACC_WARNING
        AV2P_FN_HOST_ACC constexpr auto operator()(T const& lhs, T const& rhs) const
        {
            return (lhs < rhs) ? lhs : rhs;
        }
    };

    template<typename T>
    struct max
    {
        AV2P_NO_HOST_ACC_WARNING
        AV2P_FN_HOST_ACC constexpr auto operator()(T const& lhs, T const& rhs) const
        {
            return (lhs > rhs) ? lhs : rhs;
        }
    };
} // namespace aV2P::meta

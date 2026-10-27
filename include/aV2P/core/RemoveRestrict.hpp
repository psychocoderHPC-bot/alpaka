/* Copyright 2021 Rene Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Config.hpp"

namespace aV2P
{
    //! Removes __restrict__ from a type
    template<typename T>
    struct remove_restrict
    {
        using type = T;
    };

#if AV2P_COMP_MSVC
    template<typename T>
    struct remove_restrict<T* __restrict>
    {
        using type = T*;
    };
#else
    template<typename T>
    struct remove_restrict<T* __restrict__>
    {
        using type = T*;
    };
#endif

    //! Helper to remove __restrict__ from a type
    template<typename T>
    using remove_restrict_t = typename remove_restrict<T>::type;
} // namespace aV2P

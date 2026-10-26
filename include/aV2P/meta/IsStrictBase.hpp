/* Copyright 2022 Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <type_traits>

namespace aV2P::meta
{
    //! The trait is true if TDerived is derived from TBase but is not TBase itself.
    template<typename TBase, typename TDerived>
    using IsStrictBase = std::
        integral_constant<bool, std::is_base_of_v<TBase, TDerived> && !std::is_same_v<TBase, std::decay_t<TDerived>>>;
} // namespace aV2P::meta

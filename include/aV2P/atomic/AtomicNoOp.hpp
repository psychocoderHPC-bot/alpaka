/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/atomic/Traits.hpp"

namespace aV2P
{
    //! The NoOp atomic ops.
    class AtomicNoOp
    {
    };

    namespace trait
    {
        //! The NoOp atomic operation.
        template<typename TOp, typename T, typename THierarchy>
        struct AtomicOp<TOp, AtomicNoOp, T, THierarchy>
        {
            AV2P_FN_HOST static auto atomicOp(AtomicNoOp const& /* atomic */, T* const addr, T const& value) -> T
            {
                return TOp()(addr, value);
            }

            AV2P_FN_HOST static auto atomicOp(
                AtomicNoOp const& /* atomic */,
                T* const addr,
                T const& compare,
                T const& value) -> T
            {
                return TOp()(addr, compare, value);
            }
        };
    } // namespace trait
} // namespace aV2P

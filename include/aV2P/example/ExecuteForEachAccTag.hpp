/* Copyright 2023 Jeffrey Kelling, Bernhard Manfred Gruber, Jan Stephan, Aurora Perego, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include "aV2P/alpaka.hpp"

#include <functional>
#include <tuple>
#include <utility>

#pragma once

namespace aV2P
{
    //! execute a callable for each active accelerator tag
    //
    // @param callable callable which can be invoked with an accelerator tag
    // @return disjunction of all invocation results
    //
    template<typename TCallable>
    inline auto executeForEachAccTag(TCallable&& callable)
    {
        // Execute the callable once for each enabled accelerator.
        // Pass the tag as first argument to the callable.
        return std::apply([=](auto const&... tags) { return (callable(tags) || ...); }, aV2P::EnabledAccTags{});
    }
} // namespace aV2P

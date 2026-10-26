/* Copyright 2022 René Widera, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Common.hpp"
#include "aV2P/core/Config.hpp"

#include <new>

namespace aV2P::core
{
    AV2P_FN_INLINE AV2P_FN_HOST auto alignedAlloc(size_t alignment, size_t size) -> void*
    {
        if(size == 0)
        {
            return nullptr;
        }
        else
        {
            return ::operator new(size, std::align_val_t{alignment});
        }
    }

    AV2P_FN_INLINE AV2P_FN_HOST void alignedFree(size_t alignment, void* ptr)
    {
        ::operator delete(ptr, std::align_val_t{alignment});
    }
} // namespace aV2P::core

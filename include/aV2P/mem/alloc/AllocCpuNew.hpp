/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Common.hpp"
#include "aV2P/mem/alloc/Traits.hpp"

namespace aV2P
{
    //! The CPU new allocator.
    class AllocCpuNew : public interface::Implements<ConceptMemAlloc, AllocCpuNew>
    {
    };

    namespace trait
    {
        //! The CPU new allocator memory allocation trait specialization.
        template<typename T>
        struct Malloc<T, AllocCpuNew>
        {
            AV2P_FN_HOST static auto malloc(AllocCpuNew const& /* alloc */, std::size_t const& sizeElems) -> T*
            {
                return new T[sizeElems];
            }
        };

        //! The CPU new allocator memory free trait specialization.
        template<typename T>
        struct Free<T, AllocCpuNew>
        {
            AV2P_FN_HOST static auto free(AllocCpuNew const& /* alloc */, T const* const ptr) -> void
            {
                return delete[] ptr;
            }
        };
    } // namespace trait
} // namespace aV2P

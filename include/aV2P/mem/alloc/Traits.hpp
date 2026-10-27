/* Copyright 2022 Benjamin Worpitz, Matthias Werner, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Common.hpp"
#include "aV2P/core/Interface.hpp"
#include "aV2P/dev/Traits.hpp"
#include "aV2P/dim/Traits.hpp"
#include "aV2P/extent/Traits.hpp"

namespace aV2P
{
    struct ConceptMemAlloc
    {
    };

    //! The allocator traits.
    namespace trait
    {
        //! The memory allocation trait.
        template<typename T, typename TAlloc, typename TSfinae = void>
        struct Malloc;

        //! The memory free trait.
        template<typename T, typename TAlloc, typename TSfinae = void>
        struct Free;
    } // namespace trait

    //! \return The pointer to the allocated memory.
    template<typename T, typename TAlloc>
    AV2P_FN_HOST auto malloc(TAlloc const& alloc, std::size_t const& sizeElems) -> T*
    {
        using ImplementationBase = interface::ImplementationBase<ConceptMemAlloc, TAlloc>;
        return trait::Malloc<T, ImplementationBase>::malloc(alloc, sizeElems);
    }

    //! Frees the memory identified by the given pointer.
    template<typename TAlloc, typename T>
    AV2P_FN_HOST auto free(TAlloc const& alloc, T const* const ptr) -> void
    {
        using ImplementationBase = interface::ImplementationBase<ConceptMemAlloc, TAlloc>;
        trait::Free<T, ImplementationBase>::free(alloc, ptr);
    }
} // namespace aV2P

/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/block/sync/Traits.hpp"
#include "aV2P/core/Common.hpp"

namespace aV2P
{
    //! The no op block synchronization.
    class BlockSyncNoOp : public interface::Implements<ConceptBlockSync, BlockSyncNoOp>
    {
    };

    namespace trait
    {
        template<>
        struct SyncBlockThreads<BlockSyncNoOp>
        {
            AV2P_NO_HOST_ACC_WARNING
            AV2P_FN_ACC static auto syncBlockThreads(BlockSyncNoOp const& /* blockSync */) -> void
            {
                // Nothing to do.
            }
        };

        template<typename TOp>
        struct SyncBlockThreadsPredicate<TOp, BlockSyncNoOp>
        {
            AV2P_NO_HOST_ACC_WARNING
            AV2P_FN_ACC static auto syncBlockThreadsPredicate(BlockSyncNoOp const& /* blockSync */, int predicate)
                -> int
            {
                return predicate;
            }
        };
    } // namespace trait
} // namespace aV2P

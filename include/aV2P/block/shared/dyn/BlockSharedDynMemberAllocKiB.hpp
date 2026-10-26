/* Copyright 2022 Jeffrey Kelling
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <cstdint>

namespace aV2P
{
#ifndef AV2P_BLOCK_SHARED_DYN_MEMBER_ALLOC_KIB
#    define AV2P_BLOCK_SHARED_DYN_MEMBER_ALLOC_KIB 47u
#endif
    constexpr std::uint32_t BlockSharedDynMemberAllocKiB = AV2P_BLOCK_SHARED_DYN_MEMBER_ALLOC_KIB;
} // namespace aV2P

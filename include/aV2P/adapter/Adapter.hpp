/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

//! Single entry point for the aV2P -> alpaka 3 adapter layer.
//!
//! Translation between the legacy 2.2 concepts and alpaka 3 is centralized here:
//!  - WorkDiv.hpp    : aV2P::Vec <-> alpaka::Vec and WorkDiv -> ThreadSpec
//!  - Mapping.hpp    : legacy accelerator tag -> api/deviceKind/exec
//!  - Acc.hpp        : accelerator facade and legacy accelerator aliases
//!  - Dev.hpp        : device wrapper
//!  - Platform.hpp   : platform wrapper
//!  - Queue.hpp      : queue wrapper (blocking/non-blocking)
//!  - Event.hpp      : event wrapper
//!  - Kernel.hpp     : kernel task + launch
//!  - AccTraits.hpp  : legacy accelerator traits / kernel-side index and workdiv bridge
//!  - Buf.hpp        : owned buffer wrapper
//!  - BufTraits.hpp  : legacy buffer traits and allocators
//!  - MemOps.hpp     : copy / fill / memset tasks

#include "aV2P/adapter/Acc.hpp"
#include "aV2P/adapter/AccTraits.hpp"
#include "aV2P/adapter/Atomic.hpp"
#include "aV2P/adapter/Barrier.hpp"
#include "aV2P/adapter/Buf.hpp"
#include "aV2P/adapter/BufTraits.hpp"
#include "aV2P/adapter/Dev.hpp"
#include "aV2P/adapter/Event.hpp"
#include "aV2P/adapter/Intrinsic.hpp"
#include "aV2P/adapter/Kernel.hpp"
#include "aV2P/adapter/Mapping.hpp"
#include "aV2P/adapter/Math.hpp"
#include "aV2P/adapter/MemFence.hpp"
#include "aV2P/adapter/MemOps.hpp"
#include "aV2P/adapter/Platform.hpp"
#include "aV2P/adapter/Queue.hpp"
#include "aV2P/adapter/Rand.hpp"
#include "aV2P/adapter/SharedMem.hpp"
#include "aV2P/adapter/Warp.hpp"
#include "aV2P/adapter/WorkDiv.hpp"

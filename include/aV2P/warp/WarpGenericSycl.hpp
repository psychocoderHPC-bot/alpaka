/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// Compatibility header: the legacy per-backend warp implementation classes were removed when the
// backends were replaced by the alpaka 3 adapters. The warp entry points and traits live in
// `aV2P/warp/Traits.hpp` and `aV2P/adapter/Warp.hpp`; this header keeps legacy includes compiling.
#include "aV2P/warp/Traits.hpp"

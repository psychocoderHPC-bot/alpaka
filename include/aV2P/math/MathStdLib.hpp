/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// Compatibility header: the legacy `MathStdLib` implementation classes were removed when the backend
// implementations were replaced by the alpaka 3 adapters. The math entry points, their traits and the
// ADL customization points live in `aV2P/math/Traits.hpp`; including this header keeps legacy source
// that only included it compiling.
#include "aV2P/math/Traits.hpp"

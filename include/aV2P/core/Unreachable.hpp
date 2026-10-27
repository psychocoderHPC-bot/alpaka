/* Copyright 2025 Jan Stephan, Jeffrey Kelling, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Config.hpp"

//! Before CUDA 11.5 nvcc is unable to correctly identify return statements in 'if constexpr' branches. It will issue
//! a false warning about a missing return statement unless it is told that the following code section is unreachable.
//!
//! \param x A dummy value for the expected return type of the calling function.
#if(AV2P_COMP_NVCC && AV2P_ARCH_PTX)
#    define AV2P_UNREACHABLE(...) __builtin_unreachable()
#elif AV2P_COMP_MSVC
#    define AV2P_UNREACHABLE(...) __assume(false)
#elif AV2P_COMP_GNUC || AV2P_COMP_CLANG
#    define AV2P_UNREACHABLE(...) __builtin_unreachable()
#else
#    define AV2P_UNREACHABLE(...)
#endif

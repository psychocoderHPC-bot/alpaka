/* Copyright 2021 Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Config.hpp"

//! Suggests unrolling of the directly following loop to the compiler.
//!
//! Usage:
//!  `AV2P_UNROLL
//!  for(...){...}`
// \TODO: Implement for other compilers.
#if AV2P_ARCH_PTX
#    define AV2P_UNROLL_STRINGIFY(x) #x
#    define AV2P_UNROLL(...) _Pragma(AV2P_UNROLL_STRINGIFY(unroll __VA_ARGS__))
#elif AV2P_COMP_IBM
#    define AV2P_UNROLL_STRINGIFY(x) #x
#    define AV2P_UNROLL(...) _Pragma(AV2P_UNROLL_STRINGIFY(unroll(__VA_ARGS__)))
#elif AV2P_COMP_PGI
#    define AV2P_UNROLL(...) _Pragma("unroll")
#else
#    define AV2P_UNROLL(...)
#endif

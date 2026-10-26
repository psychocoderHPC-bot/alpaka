/* Copyright 2023 Jan Stephan, Luca Ferragina, Aurora Perego, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

//! SYCL device-printf shim.
//!
//! The 2.2 `core/Sycl.hpp` mixed three concerns: the SYCL vector-type traits, the device `printf`
//! shim, and the `AlpakaFormat` type.  The vector traits were part of the deleted SYCL backend
//! implementation; only the printf shim survives because `AV2P_THROW_ACC` uses it in device code.
//!
//! In a non-SYCL build this header is empty.

#include "aV2P/core/Config.hpp"

#include <cstdio> // the #define printf(...) breaks <cstdio> if it is included afterwards

#ifdef AV2P_ACC_SYCL_ENABLED

#    include <sycl/sycl.hpp>

// if SYCL is enabled with the AMD backend the printf will be killed because of missing compiler support
#    ifdef __AMDGCN__
#        define printf(...)
#    else

#        ifdef __SYCL_DEVICE_ONLY__
using AlpakaFormat = char const* [[clang::opencl_constant]];
#        else
using AlpakaFormat = char const*;
#        endif

#        if AV2P_COMP_CLANG
#            pragma clang diagnostic push
#            pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#        endif

#        define printf(FORMAT, ...)                                                                                   \
            do                                                                                                        \
            {                                                                                                         \
                static auto const format = AlpakaFormat{FORMAT};                                                      \
                sycl::ext::oneapi::experimental::printf(format, ##__VA_ARGS__);                                       \
            } while(false)

#        if AV2P_COMP_CLANG
#            pragma clang diagnostic pop
#        endif

#    endif
#endif

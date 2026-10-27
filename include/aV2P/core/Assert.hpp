/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Common.hpp"

#include <cassert>
#include <type_traits>

//! The assert can be explicit disabled by defining NDEBUG
#define AV2P_ASSERT(...) assert(__VA_ARGS__)

//! Macro which expands to a noop.
//! Macro enforces an semicolon after the call.
#define AV2P_NOOP(...)                                                                                              \
    do                                                                                                                \
    {                                                                                                                 \
    } while(false)

//! AV2P_ASSERT_ACC_IMPL is an assert-like macro.
//! It can be disabled setting the AV2P_DISABLE_ASSERT_ACC preprocessor symbol or the NDEBUG preprocessor symbol.
#if !defined(AV2P_DISABLE_ASSERT_ACC)
#    define AV2P_ASSERT_ACC_IMPL(...) AV2P_ASSERT(__VA_ARGS__)
#else
#    define AV2P_ASSERT_ACC_IMPL(...) AV2P_NOOP(__VA_ARGS__)
#endif

//! AV2P_ASSERT_ACC is an assert-like macro.
//!
//! In device code for a GPU or SYCL backend it can be disabled setting the AV2P_DISABLE_ASSERT_ACC preprocessor
//! symbol or the NDEBUG preprocessor symbol. In device code for a native C++ CPU backend and in host code, it is
//! equivalent to AV2P_ASSERT, and can be disabled setting the NDEBUG preprocessor symbol.
#if defined(AV2P_ACC_GPU_CUDA_ENABLED) && defined(__CUDA_ARCH__)
// CUDA device code
#    define AV2P_ASSERT_ACC(...) AV2P_ASSERT_ACC_IMPL(__VA_ARGS__)
#elif defined(AV2P_ACC_GPU_HIP_ENABLED) && defined(__HIP_DEVICE_COMPILE__)
// HIP/ROCm device code
#    define AV2P_ASSERT_ACC(...) AV2P_ASSERT_ACC_IMPL(__VA_ARGS__)
#elif defined(AV2P_ACC_SYCL_ENABLED) && defined(__SYCL_DEVICE_ONLY__)
// SYCL/oneAPI device code
#    if defined(SYCL_EXT_ONEAPI_ASSERT)
#        define AV2P_ASSERT_ACC(...) AV2P_ASSERT_ACC_IMPL(__VA_ARGS__)
#    else
#        define AV2P_ASSERT_ACC(...) AV2P_NOOP(__VA_ARGS__)
#    endif
// add here any other #elif conditions for non-CPU backends
// ...
#else
// CPU backend, or host code
#    define AV2P_ASSERT_ACC(...) AV2P_ASSERT(__VA_ARGS__)
#endif

namespace aV2P::core
{
    namespace detail
    {
        template<typename TArg>
        struct AssertValueUnsigned
        {
            AV2P_NO_HOST_ACC_WARNING AV2P_FN_HOST_ACC static constexpr auto assertValueUnsigned(
                [[maybe_unused]] TArg const& arg)
            {
                if constexpr(std::is_signed_v<TArg>)
                    AV2P_ASSERT_ACC(arg >= 0);

                // Nothing to do for unsigned types.
            }
        };
    } // namespace detail

    //! This method checks integral values if they are greater or equal zero.
    //! The implementation prevents warnings for checking this for unsigned types.
    AV2P_NO_HOST_ACC_WARNING
    template<typename TArg>
    AV2P_FN_HOST_ACC constexpr auto assertValueUnsigned(TArg const& arg) -> void
    {
        detail::AssertValueUnsigned<TArg>::assertValueUnsigned(arg);
    }

    namespace detail
    {
        template<typename TLhs, typename TRhs>
        struct AssertGreaterThan
        {
            AV2P_NO_HOST_ACC_WARNING AV2P_FN_HOST_ACC static constexpr auto assertGreaterThan(
                [[maybe_unused]] TRhs const& rhs)
            {
                if constexpr(std::is_signed_v<TRhs> || (TLhs::value != 0u))
                    AV2P_ASSERT_ACC(TLhs::value > rhs);

                // Nothing to do for unsigned types comparing to zero.
            }
        };
    } // namespace detail

    //! This function asserts that the integral value TLhs is greater than TRhs.
    AV2P_NO_HOST_ACC_WARNING
    template<typename TLhs, typename TRhs>
    AV2P_FN_HOST_ACC constexpr auto assertGreaterThan(TRhs const& rhs) -> void
    {
        detail::AssertGreaterThan<TLhs, TRhs>::assertGreaterThan(rhs);
    }
} // namespace aV2P::core

/* Copyright 2022 Alexander Matthes, Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Config.hpp"

#include <iostream>
#include <string>
#include <utility>

//! The no debug level.
#define AV2P_DEBUG_DISABLED 0
//! The minimal debug level.
#define AV2P_DEBUG_MINIMAL 1
//! The full debug level.
#define AV2P_DEBUG_FULL 2

#ifndef AV2P_DEBUG
//! Set the minimum log level if it is not defined.
#    define AV2P_DEBUG AV2P_DEBUG_DISABLED
#endif

namespace aV2P::core::detail
{
    //! Scope logger.
    class ScopeLogStdOut final
    {
    public:
        explicit ScopeLogStdOut(std::string sScope) : m_sScope(std::move(sScope))
        {
            std::cout << "[+] " << m_sScope << std::endl;
        }

        ScopeLogStdOut(ScopeLogStdOut const&) = delete;
        ScopeLogStdOut(ScopeLogStdOut&&) = delete;
        auto operator=(ScopeLogStdOut const&) -> ScopeLogStdOut& = delete;
        auto operator=(ScopeLogStdOut&&) -> ScopeLogStdOut& = delete;

        ~ScopeLogStdOut()
        {
            std::cout << "[-] " << m_sScope << std::endl;
        }

    private:
        std::string const m_sScope;
    };
} // namespace aV2P::core::detail

// Define AV2P_DEBUG_MINIMAL_LOG_SCOPE.
#if AV2P_DEBUG >= AV2P_DEBUG_MINIMAL
#    define AV2P_DEBUG_MINIMAL_LOG_SCOPE ::aV2P::core::detail::ScopeLogStdOut const scopeLogStdOut(__func__)
#else
#    define AV2P_DEBUG_MINIMAL_LOG_SCOPE
#endif

// Define AV2P_DEBUG_FULL_LOG_SCOPE.
#if AV2P_DEBUG >= AV2P_DEBUG_FULL
#    define AV2P_DEBUG_FULL_LOG_SCOPE ::aV2P::core::detail::ScopeLogStdOut const scopeLogStdOut(__func__)
#else
#    define AV2P_DEBUG_FULL_LOG_SCOPE
#endif

// Define AV2P_DEBUG_BREAK.
#if AV2P_DEBUG >= AV2P_DEBUG_MINIMAL
#    if AV2P_COMP_GNUC || AV2P_COMP_CLANG
#        define AV2P_DEBUG_BREAK ::__builtin_trap()
#    elif AV2P_COMP_MSVC
#        define AV2P_DEBUG_BREAK ::__debugbreak()
#    else
#        define AV2P_DEBUG_BREAK
  // #error debug-break for current compiler not implemented!
#    endif
#else
#    define AV2P_DEBUG_BREAK
#endif

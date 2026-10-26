/* Copyright 2022 Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Interface.hpp"
#include "aV2P/dev/DevCpu.hpp"
#include "aV2P/platform/Traits.hpp"

#include <sstream>
#include <vector>

namespace aV2P
{
    //! The CPU device platform.
    struct PlatformCpu : interface::Implements<ConceptPlatform, PlatformCpu>
    {
#if defined(AV2P_COMP_GNUC) && AV2P_COMP_GNUC >= AV2P_VERSION_NUMBER(11, 0, 0)                                  \
    && AV2P_COMP_GNUC < AV2P_VERSION_NUMBER(12, 0, 0)
        // This is a workaround for g++-11 bug: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=96295
        // g++-11 complains in *all* places where a PlatformCpu is used, that it "may be used uninitialized"
        char c = {};
#endif
    };

    namespace trait
    {
        //! The CPU device device type trait specialization.
        template<>
        struct DevType<PlatformCpu>
        {
            using type = DevCpu;
        };

        //! The CPU platform device count get trait specialization.
        template<>
        struct GetDevCount<PlatformCpu>
        {
            AV2P_FN_HOST static auto getDevCount(PlatformCpu const&) -> std::size_t
            {
                AV2P_DEBUG_FULL_LOG_SCOPE;

                return 1;
            }
        };

        //! The CPU platform device get trait specialization.
        template<>
        struct GetDevByIdx<PlatformCpu>
        {
            AV2P_FN_HOST static auto getDevByIdx(PlatformCpu const& platform, std::size_t const& devIdx) -> DevCpu
            {
                AV2P_DEBUG_FULL_LOG_SCOPE;

                std::size_t const devCount = getDevCount(platform);
                if(devIdx >= devCount)
                {
                    std::stringstream ssErr;
                    ssErr << "Unable to return device handle for CPU device with index " << devIdx
                          << " because there are only " << devCount << " devices!";
                    throw std::runtime_error(ssErr.str());
                }

                return {};
            }
        };
    } // namespace trait
} // namespace aV2P

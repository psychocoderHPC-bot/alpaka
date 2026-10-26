/* Copyright 2022 Benjamin Worpitz, Rene Widera, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/dev/DevCpu.hpp"
#include "aV2P/event/EventCpu.hpp"
#include "aV2P/wait/Traits.hpp"

namespace aV2P::trait
{
    //! The CPU device thread wait specialization.
    //!
    //! Blocks until the device has completed all preceding requested tasks.
    //! Tasks that are enqueued or queues that are created after this call is made are not waited for.
    template<>
    struct CurrentThreadWaitFor<DevCpu>
    {
        AV2P_FN_HOST static auto currentThreadWaitFor(DevCpu const& dev) -> void
        {
            AV2P_DEBUG_FULL_LOG_SCOPE;

            generic::currentThreadWaitForDevice(dev);
        }
    };
} // namespace aV2P::trait

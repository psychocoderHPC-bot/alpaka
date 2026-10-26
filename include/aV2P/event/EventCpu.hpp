/* Copyright 2020 Jeffrey Kelling, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/dev/DevCpu.hpp"
#include "aV2P/event/EventGenericThreads.hpp"

namespace aV2P
{
    using EventCpu = EventGenericThreads<DevCpu>;
} // namespace aV2P

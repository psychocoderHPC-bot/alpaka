/* Copyright 2020 Jeffrey Kelling, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/event/EventCpu.hpp"
#include "aV2P/queue/QueueGenericThreadsBlocking.hpp"

namespace aV2P
{
    using QueueCpuBlocking = QueueGenericThreadsBlocking<DevCpu>;
} // namespace aV2P

/* Copyright 2020 Jeffrey Kelling, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/event/EventCpu.hpp"
#include "aV2P/queue/QueueGenericThreadsNonBlocking.hpp"

namespace aV2P
{
    using QueueCpuNonBlocking = QueueGenericThreadsNonBlocking<DevCpu>;
} // namespace aV2P

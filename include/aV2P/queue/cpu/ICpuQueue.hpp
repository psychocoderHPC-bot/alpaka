/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/dev/DevCpu.hpp"
#include "aV2P/queue/cpu/IGenericThreadsQueue.hpp"

namespace aV2P::cpu
{
    //! The CPU queue interface
    using ICpuQueue = IGenericThreadsQueue<DevCpu>;
} // namespace aV2P::cpu

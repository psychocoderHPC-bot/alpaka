/* Copyright 2022 Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/alpaka.hpp"
#include "aV2P/core/DemangleTypeNames.hpp"

#include <chrono>
#include <type_traits>
#include <utility>

namespace aV2P::test::integ
{
    //! Measures and returns the runtime in ms of the passed callable.
    //! \param callable An object with operator().
    template<typename TCallable>
    auto measureRunTimeMs(TCallable&& callable) -> std::chrono::milliseconds::rep
    {
        auto const start = std::chrono::high_resolution_clock::now();
        std::forward<TCallable>(callable)();
        auto const end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }

    //! \return The run time of the given kernel.
    template<typename TQueue, typename TTask>
    auto measureTaskRunTimeMs(TQueue& queue, TTask&& task) -> std::chrono::milliseconds::rep
    {
#if AV2P_DEBUG >= AV2P_DEBUG_MINIMAL
        std::cout << "measureKernelRunTime("
                  << " queue: " << core::demangled<TQueue> << " task: " << core::demangled<std::decay_t<TTask>> << ")"
                  << std::endl;
#endif
        // Wait for the queue to finish all tasks enqueued prior to the given task.
        aV2P::wait(queue);

        return measureRunTimeMs(
            [&]
            {
                aV2P::enqueue(queue, std::forward<TTask>(task));

                // Wait for the queue to finish the task execution to measure its run time.
                aV2P::wait(queue);
            });
    }
} // namespace aV2P::test::integ

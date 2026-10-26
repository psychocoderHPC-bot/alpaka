/* Copyright 2020 Benjamin Worpitz, Matthias Werner, Jakob Krude, Sergei Bastrakov, Bernhard Manfred Gruber,
 * Tapish Narwal
 * SPDX-License-Identifier: ISC
 */

#include "BoundaryKernel.hpp"
#include "StencilKernel.hpp"
#include "analyticalSolution.hpp"

#ifdef PNGWRITER_ENABLED
#    include "writeImage.hpp"
#endif

#include <aV2P/alpaka.hpp>
#include <aV2P/example/ExecuteForEachAccTag.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <utility>

//! Each kernel computes the next step for one point.
//! Therefore the number of threads should be equal to numNodesX.
//! Every time step the kernel will be executed numNodesX-times
//! After every step the curr-buffer will be set to the calculated values
//! from the next-buffer.
//!
//! In standard projects, you typically do not execute the code with any available accelerator.
//! Instead, a single accelerator is selected once from the active accelerators and the kernels are executed with the
//! selected accelerator only. If you use the example as the starting point for your project, you can rename the
//! example() function to main() and move the accelerator tag to the function body.
template<aV2P::concepts::Tag TAccTag>
auto example(TAccTag const&) -> int
{
    // Set Dim and Idx type
    using Dim = aV2P::DimInt<2u>;
    using Idx = uint32_t;

    // Define the accelerator
    using Acc = aV2P::TagToAcc<TAccTag, Dim, Idx>;
    std::cout << "Using alpaka accelerator: " << aV2P::getAccName<Acc>() << std::endl;

    // Select specific devices
    auto const platformHost = aV2P::PlatformCpu{};
    auto const devHost = aV2P::getDevByIdx(platformHost, 0);
    auto const platformAcc = aV2P::Platform<Acc>{};
    // get suitable device for this Acc
    auto const devAcc = aV2P::getDevByIdx(platformAcc, 0);

    // simulation defines
    // {Y, X}
    constexpr aV2P::Vec<Dim, Idx> numNodes{64, 64};
    constexpr aV2P::Vec<Dim, Idx> haloSize{2, 2};
    constexpr aV2P::Vec<Dim, Idx> extent = numNodes + haloSize;

    constexpr uint32_t numTimeSteps = 4000;
    constexpr double tMax = 0.1;

    // x, y in [0, 1], t in [0, tMax]
    constexpr double dx = 1.0 / static_cast<double>(extent[1] - 1);
    constexpr double dy = 1.0 / static_cast<double>(extent[0] - 1);
    constexpr double dt = tMax / static_cast<double>(numTimeSteps);

    // Check the stability condition
    double r = 2 * dt / ((dx * dx * dy * dy) / (dx * dx + dy * dy));
    if(r > 1.)
    {
        std::cerr << "Stability condition check failed: dt/min(dx^2,dy^2) = " << r
                  << ", it is required to be <= 0.5\n";
        return EXIT_FAILURE;
    }

    // Initialize host-buffer
    // This buffer will hold the current values (used for the next step)
    auto uBufHost = aV2P::allocBuf<double, Idx>(devHost, extent);

    // Accelerator buffer
    auto uCurrBufAcc = aV2P::allocBuf<double, Idx>(devAcc, extent);
    auto uNextBufAcc = aV2P::allocBuf<double, Idx>(devAcc, extent);

    auto const pitchCurrAcc{aV2P::getPitchesInBytes(uCurrBufAcc)};
    auto const pitchNextAcc{aV2P::getPitchesInBytes(uNextBufAcc)};

    // Set buffer to initial conditions
    initalizeBuffer(uBufHost, dx, dy);

    // Select queue
    using QueueProperty = aV2P::NonBlocking;
    using QueueAcc = aV2P::Queue<Acc, QueueProperty>;
    QueueAcc dumpQueue{devAcc};
    QueueAcc computeQueue{devAcc};

    // Copy host -> device
    aV2P::memcpy(computeQueue, uCurrBufAcc, uBufHost);
    aV2P::wait(computeQueue);

    // Define a workdiv for the given problem
    constexpr aV2P::Vec<Dim, Idx> elemPerThread{1, 1};

    // Appropriate chunk size to split your problem for your Acc
    constexpr Idx xSize = 16u;
    constexpr Idx ySize = 16u;
    constexpr Idx halo = 2u;
    constexpr aV2P::Vec<Dim, Idx> chunkSize{ySize, xSize};
    constexpr auto sharedMemSize = (ySize + halo) * (xSize + halo);

    constexpr aV2P::Vec<Dim, Idx> numChunks{
        aV2P::core::divCeil(numNodes[0], chunkSize[0]),
        aV2P::core::divCeil(numNodes[1], chunkSize[1]),
    };

    assert(
        numNodes[0] % chunkSize[0] == 0 && numNodes[1] % chunkSize[1] == 0
        && "Domain must be divisible by chunk size");

    StencilKernel<sharedMemSize> stencilKernel;
    BoundaryKernel boundaryKernel;

    // Get max threads that can be run in a block for this kernel
    auto const kernelFunctionAttributes = aV2P::getFunctionAttributes<Acc>(
        devAcc,
        stencilKernel,
        uCurrBufAcc.data(),
        uNextBufAcc.data(),
        chunkSize,
        pitchCurrAcc,
        pitchNextAcc,
        dx,
        dy,
        dt);
    auto const maxThreadsPerBlock = kernelFunctionAttributes.maxThreadsPerBlock;

    auto const threadsPerBlock
        = maxThreadsPerBlock < chunkSize.prod() ? aV2P::Vec<Dim, Idx>{maxThreadsPerBlock, 1} : chunkSize;

    aV2P::WorkDivMembers<Dim, Idx> workDiv_manual{numChunks, threadsPerBlock, elemPerThread};

    // Simulate
    for(uint32_t step = 1; step <= numTimeSteps; ++step)
    {
        // Compute next values
        aV2P::exec<Acc>(
            computeQueue,
            workDiv_manual,
            stencilKernel,
            uCurrBufAcc.data(),
            uNextBufAcc.data(),
            chunkSize,
            pitchCurrAcc,
            pitchNextAcc,
            dx,
            dy,
            dt);

        // Apply boundaries
        aV2P::exec<Acc>(
            computeQueue,
            workDiv_manual,
            boundaryKernel,
            uNextBufAcc.data(),
            chunkSize,
            pitchNextAcc,
            step,
            dx,
            dy,
            dt);

#ifdef PNGWRITER_ENABLED
        if((step - 1) % 100 == 0)
        {
            aV2P::wait(computeQueue);
            aV2P::memcpy(dumpQueue, uBufHost, uCurrBufAcc);
            aV2P::wait(dumpQueue);
            writeImage(step - 1, uBufHost);
        }
#endif

        // So we just swap next and curr (shallow copy)
        std::swap(uNextBufAcc, uCurrBufAcc);
    }

    // Copy device -> host
    aV2P::wait(computeQueue);
    aV2P::memcpy(dumpQueue, uBufHost, uCurrBufAcc);
    aV2P::wait(dumpQueue);

    // Validate
    auto const [resultIsCorrect, maxError] = validateSolution(uBufHost, extent, dx, dy, tMax);

    if(resultIsCorrect)
    {
        std::cout << "Execution results correct!" << std::endl;
        return EXIT_SUCCESS;
    }
    else
    {
        std::cout << "Execution results incorrect: Max error = " << maxError << " (the grid resolution may be too low)"
                  << std::endl;
        return EXIT_FAILURE;
    }
}

auto main() -> int
{
    std::cout << "Check enabled accelerator tags:" << std::endl;
    aV2P::printTagNames<aV2P::EnabledAccTags>();
    // Execute the example once for each enabled accelerator.
    // If you would like to execute it for a single accelerator only you can use the following code.
    //  \code{.cpp}
    //  auto tag = TagCpuSerial;
    //  return example(tag);
    //  \endcode
    //
    // valid tags:
    //   TagCpuSerial, TagGpuHipRt, TagGpuCudaRt, TagCpuOmp2Blocks, TagCpuTbbBlocks,
    //   TagCpuOmp2Threads, TagCpuSycl, TagCpuTbbBlocks, TagCpuThreads,
    //   TagFpgaSyclIntel, TagGenericSycl, TagGpuSyclIntel
    return aV2P::executeForEachAccTag([=](auto const& tag) { return example(tag); });
}

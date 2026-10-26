/* Copyright 2020 Benjamin Worpitz, Sergei Bastrakov, Jakob Krude, Bernhard Manfred Gruber
 * SPDX-License-Identifier: ISC
 */

#include <aV2P/alpaka.hpp>
#include <aV2P/example/ExecuteForEachAccTag.hpp>

#include <cstdint>
#include <cstdlib>
#include <iostream>

//! This functor defines the function for which the integral is to be computed.
struct Function
{
    //! \tparam TAcc The accelerator environment to be executed on.
    //! \param acc The accelerator to be executed on.
    //! \param x The argument.
    template<typename TAcc>
    AV2P_FN_ACC auto operator()(TAcc const& acc, float const x) -> float
    {
        return aV2P::math::sqrt(acc, (1.0f - x * x));
    }
};

//! The kernel executing the parallel logic.
//! Each Thread generates X pseudo random numbers and compares them with the given function.
//! The local result will be added to a global result.
struct Kernel
{
    //! The kernel entry point.
    //! \tparam TAcc The accelerator environment to be executed on.
    //! \tparam TFunctor A wrapper for a function.
    //! \param acc The accelerator to be executed on.
    //! \param numPoints The total number of points to be calculated.
    //! \param globalCounter The sum of all local results.
    //! \param functor The function for which the integral is to be computed.
    AV2P_NO_HOST_ACC_WARNING
    template<typename TAcc, typename TFunctor>
    AV2P_FN_ACC auto operator()(
        TAcc const& acc,
        size_t const numPoints,
        uint32_t* const globalCounter,
        TFunctor functor) const -> void
    {
        // Get the global linearized thread idx.
        auto const globalThreadIdx = aV2P::getIdx<aV2P::Grid, aV2P::Threads>(acc);
        auto const globalThreadExtent = aV2P::getWorkDiv<aV2P::Grid, aV2P::Threads>(acc);

        auto const linearizedGlobalThreadIdx = aV2P::mapIdx<1u>(globalThreadIdx, globalThreadExtent)[0];
        // Setup generator engine and distribution.
        auto engine = aV2P::rand::engine::createDefault(
            acc,
            linearizedGlobalThreadIdx,
            0); // No specific subsequence start.
        // For simplicity the interval is fixed to [0.0,1.0].
        auto dist = aV2P::rand::distribution::createUniformReal<float>(acc);

        uint32_t localCount = 0;
        for(size_t i = linearizedGlobalThreadIdx; i < numPoints; i += globalThreadExtent.prod())
        {
            // Generate a point in the 2D interval.
            float x = dist(engine);
            float y = dist(engine);
            // Count every time where the point is "below" the given function.
            if(y <= functor(acc, x))
            {
                ++localCount;
            }
        }

        // Add the local result to the sum of the other results.
        aV2P::atomicAdd(acc, globalCounter, localCount, aV2P::hierarchy::Blocks{});
    }
};

// In standard projects, you typically do not execute the code with any available accelerator.
// Instead, a single accelerator is selected once from the active accelerators and the kernels are executed with the
// selected accelerator only. If you use the example as the starting point for your project, you can rename the
// example() function to main() and move the accelerator tag to the function body.
template<aV2P::concepts::Tag TAccTag>
auto example(TAccTag const&) -> int
{
    // Defines and setup.
    using Dim = aV2P::DimInt<1>;
    using Idx = std::size_t;
    using Vec = aV2P::Vec<Dim, Idx>;
    using Acc = aV2P::TagToAcc<TAccTag, Dim, Idx>;
    using Host = aV2P::DevCpu;
    auto const platformHost = aV2P::PlatformCpu{};
    auto const devHost = aV2P::getDevByIdx(platformHost, 0);
    auto const platformAcc = aV2P::Platform<Acc>{};
    auto const devAcc = aV2P::getDevByIdx(platformAcc, 0);
    using QueueProperty = aV2P::Blocking;
    using QueueAcc = aV2P::Queue<Acc, QueueProperty>;
    QueueAcc queue{devAcc};

    using BufHost = aV2P::Buf<Host, uint32_t, Dim, Idx>;
    using BufAcc = aV2P::Buf<Acc, uint32_t, Dim, Idx>;

    // Problem parameter.
    constexpr size_t numPoints = 1'000'000u;
    constexpr size_t extent = 1u;
    constexpr size_t numThreads = 100u; // Kernel will decide numCalcPerThread.
    constexpr size_t numAlpakaElementsPerThread = 1;

    // Setup buffer.
    BufHost bufHost{aV2P::allocBuf<uint32_t, Idx>(devHost, extent)};
    BufAcc bufAcc{aV2P::allocBuf<uint32_t, Idx>(devAcc, extent)};
    uint32_t* const ptrBufAcc{std::data(bufAcc)};

    // Initialize the global count to 0.
    bufHost[0] = 0.0f;
    aV2P::memcpy(queue, bufAcc, bufHost);

    aV2P::KernelCfg<Acc> const kernelCfg = {Vec(numThreads), Vec(numAlpakaElementsPerThread)};
    Kernel kernel;

    // Let alpaka calculate good block and grid sizes given our full problem extent
    auto const workDiv = aV2P::getValidWorkDiv(kernelCfg, devAcc, kernel, numPoints, ptrBufAcc, Function{});

    aV2P::exec<Acc>(queue, workDiv, kernel, numPoints, ptrBufAcc, Function{});
    aV2P::memcpy(queue, bufHost, bufAcc);
    aV2P::wait(queue);

    // Check the result.
    uint32_t globalCount = bufHost[0];

    // Final result.
    float finalResult = globalCount / static_cast<float>(numPoints);
    constexpr double pi = 3.14159265358979323846;
    constexpr double exactResult = pi / 4.0;
    auto const error = std::abs(finalResult - exactResult);

    std::cout << "exact result (pi / 4): " << pi / 4.0 << "\n";
    std::cout << "final result: " << finalResult << "\n";
    std::cout << "error: " << error << "\n";
    return error > 0.001 ? EXIT_FAILURE : EXIT_SUCCESS;
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

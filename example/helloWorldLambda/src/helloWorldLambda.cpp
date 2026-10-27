/* Copyright 2023 Benjamin Worpitz, Erik Zenker, Bernhard Manfred Gruber, Jan Stephan
 * SPDX-License-Identifier: ISC
 */

#include <aV2P/alpaka.hpp>
#include <aV2P/example/ExecuteForEachAccTag.hpp>

#include <functional>

//! This functions says hi to the world and
//! can be called from within a kernel function.
//! It might be useful when it is necessary
//! to lift an existing function into a device
//! function.
template<typename TAcc>
void AV2P_FN_ACC hiWorldFunction(TAcc const& acc, size_t const nExclamationMarks)
{
    using Dim = aV2P::Dim<TAcc>;
    using Idx = aV2P::Idx<TAcc>;
    using Vec = aV2P::Vec<Dim, Idx>;
    using Vec1 = aV2P::Vec<aV2P::DimInt<1u>, Idx>;

    Vec const globalThreadIdx = aV2P::getIdx<aV2P::Grid, aV2P::Threads>(acc);
    Vec const globalThreadExtent = aV2P::getWorkDiv<aV2P::Grid, aV2P::Threads>(acc);
    Vec1 const linearizedGlobalThreadIdx = aV2P::mapIdx<1u>(globalThreadIdx, globalThreadExtent);

    printf(
        "[z:%u, y:%u, x:%u][linear:%u] Hi world from a function",
        static_cast<unsigned>(globalThreadIdx[0]),
        static_cast<unsigned>(globalThreadIdx[1]),
        static_cast<unsigned>(globalThreadIdx[2]),
        static_cast<unsigned>(linearizedGlobalThreadIdx[0]));

    for(size_t i = 0; i < nExclamationMarks; ++i)
    {
        printf("!");
    }

    printf("\n");
}

//! Kernel writing the Hello World message.
struct HelloWorldKernel
{
    template<typename TAcc>
    AV2P_FN_ACC void operator()(TAcc const& acc, size_t const nExclamationMarksAsArg) const
    {
        auto globalThreadIdx = aV2P::getIdx<aV2P::Grid, aV2P::Threads>(acc);
        auto globalThreadExtent = aV2P::getWorkDiv<aV2P::Grid, aV2P::Threads>(acc);
        auto linearizedGlobalThreadIdx = aV2P::mapIdx<1u>(globalThreadIdx, globalThreadExtent);

        printf(
            "[z:%u, y:%u, x:%u][linear:%u] Hello world from a lambda",
            static_cast<unsigned>(globalThreadIdx[0]),
            static_cast<unsigned>(globalThreadIdx[1]),
            static_cast<unsigned>(globalThreadIdx[2]),
            static_cast<unsigned>(linearizedGlobalThreadIdx[0]));

        for(size_t i = 0; i < nExclamationMarksAsArg; ++i)
        {
            printf("!");
        }

        printf("\n");
    }
};

// In standard projects, you typically do not execute the code with any available accelerator.
// Instead, a single accelerator is selected once from the active accelerators and the kernels are executed with the
// selected accelerator only. If you use the example as the starting point for your project, you can rename the
// example() function to main() and move the accelerator tag to the function body.
template<aV2P::concepts::Tag TAccTag>
auto example(TAccTag const&) -> int
{
// It requires support for extended lambdas when using nvcc as CUDA compiler.
// Requires sequential backend if CI is used
#if(!defined(__NVCC__) || (defined(__NVCC__) && defined(__CUDACC_EXTENDED_LAMBDA__)))                                 \
    && (!defined(AV2P_CI) || defined(AV2P_ACC_CPU_B_SEQ_T_SEQ_ENABLED))

    // Define the index domain
    using Dim = aV2P::DimInt<3>;
    using Idx = std::size_t;

    // Define the accelerator
    using Tag = TAccTag;
    using Acc = aV2P::TagToAcc<Tag, Dim, Idx>;
    std::cout << "Using alpaka accelerator: " << aV2P::getAccName<Acc>() << std::endl;

    // Defines the synchronization behavior of a queue
    //
    // choose between Blocking and NonBlocking
    using QueueProperty = aV2P::Blocking;
    using Queue = aV2P::Queue<Acc, QueueProperty>;

    // Select a device
    auto const platformAcc = aV2P::Platform<Acc>{};
    auto const devAcc = aV2P::getDevByIdx(platformAcc, 0);

    // Create a queue on the device
    Queue queue(devAcc);

    // Define the work division
    using Vec = aV2P::Vec<Dim, Idx>;
    auto const elementsPerThread = Vec::all(static_cast<Idx>(1));
    auto const elementsPerGrid = Vec{4, 2, 4};


    size_t const nExclamationMarks = 10;

    // Run the "Hello World" kernel.
    //
    // The kernel accepts the utilized accelerator as first argument.
    // Additional arguments can be provided after the kernel functor
    // declaration or stored as members of the functor.
    //
    // This example passes the number of exclamation marks that should
    // be written after we greet the world as a kernel argument.
    HelloWorldKernel kernel;

    aV2P::KernelCfg<Acc> const kernelCfg = {elementsPerGrid, elementsPerThread};

    // Let alpaka calculate good block and grid sizes given our full problem extent
    auto const workDiv = aV2P::getValidWorkDiv(kernelCfg, devAcc, kernel, nExclamationMarks);

    aV2P::exec<Tag>(queue, workDiv, kernel, nExclamationMarks);
    aV2P::wait(queue);

    return EXIT_SUCCESS;

#else
    return EXIT_SUCCESS;
#endif
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
    //   TagGenericSycl, TagGpuSyclIntel
    return aV2P::executeForEachAccTag([=](auto const& tag) { return example(tag); });
}

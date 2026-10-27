/* Copyright 2023 Sergei Bastrakov, Jan Stephan
 * SPDX-License-Identifier: ISC
 */

#include <aV2P/alpaka.hpp>
#include <aV2P/example/ExecuteForEachAccTag.hpp>

#include <cstdint>
#include <iostream>

//! Complex numbers demonstration kernel
struct ComplexKernel
{
    template<typename TAcc>
    AV2P_FN_ACC auto operator()(TAcc const& acc) const -> void
    {
        // aV2P::Complex<T> supports the same methods as std::complex<T>, they are also useable inside kernels
        auto x = aV2P::Complex<float>(0.1f, 0.2f);
        float const real = x.real();
        auto y = aV2P::Complex<float>(0.3f, 0.4f);

        // Operators are also the same
        x *= 2.0f;
        aV2P::Complex<float> z = x + y;

        // In-kernel math functions are accessed via alpaka wrappers, the same way as for real numbers
        float zAbs = aV2P::math::abs(acc, z);
    }
};

// In standard projects, you typically do not execute the code with any available accelerator.
// Instead, a single accelerator is selected once from the active accelerators and the kernels are executed with the
// selected accelerator only. If you use the example as the starting point for your project, you can rename the
// example() function to main() and move the accelerator tag to the function body.
template<aV2P::concepts::Tag TAccTag>
auto example(TAccTag const&) -> int
{
    using Idx = std::size_t;

    // Define the accelerator
    using Acc = aV2P::TagToAcc<TAccTag, aV2P::DimInt<1>, Idx>;
    std::cout << "Using alpaka accelerator: " << aV2P::getAccName<Acc>() << std::endl;

    // Defines the synchronization behavior of a queue
    using QueueProperty = aV2P::Blocking;
    using Queue = aV2P::Queue<Acc, QueueProperty>;

    // Select a device
    auto const platformAcc = aV2P::Platform<Acc>{};
    auto const devAcc = aV2P::getDevByIdx(platformAcc, 0);

    // Create a queue on the device
    Queue queue(devAcc);

    // Define the work division
    Idx const elementsPerGrid = 1u;
    Idx const elementsPerThread = 1u;

    ComplexKernel complexKernel;

    aV2P::KernelCfg<Acc> const kernelCfg = {elementsPerGrid, elementsPerThread};

    // Let alpaka calculate good block and grid sizes given our full problem extent
    auto const workDiv = aV2P::getValidWorkDiv(kernelCfg, devAcc, complexKernel);

    // Run the kernel
    aV2P::exec<Acc>(queue, workDiv, complexKernel);
    aV2P::wait(queue);

    // Usage of aV2P::Complex<T> on the host side is the same as inside kernels, except math functions are not
    // supported
    auto x = aV2P::Complex<float>(0.1f, 0.2f);
    float const real = x.real();
    auto y = aV2P::Complex<float>(0.3f, 0.4f);
    x *= 2.0f;
    aV2P::Complex<float> z = x + y;

    return EXIT_SUCCESS;
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

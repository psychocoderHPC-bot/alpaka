/* Copyright 2023 Sergei Bastrakov, Jan Stephan
 * SPDX-License-Identifier: ISC
 */

#include <aV2P/alpaka.hpp>
#include <aV2P/example/ExecuteForEachAccTag.hpp>

#include <iostream>

//! Kernel to illustrate specialization for a particular accelerator
//!
//! It has a generic operator() implementation and an overload for the CUDA accelerator.
//! When running the kernel on a CUDA device, the corresponding overload of operator() is called.
//! Otherwise the generic version is called.
//! The same technique can be applied for any function called from inside the kernel,
//! thus allowing specialization of only relevant part of the code.
//! It can be useful for optimization or accessing specific functionality not abstracted by alpaka.
//!
//! This kernel demonstrates the simplest way to achieve the effect by function overloading.
//! Note that it does not perform specialization (in C++ template meaning) of function templates.
//! We use the word "specialization" as it represents a case of having a special version for a particular accelerator.
//! One could apply a similar technique by having an additional class template parametrized by the accelerator type.
//! For such a case, both template specialization and function overloading of the methods can be employed.
struct Kernel
{
    //! Implementation for the general case
    //!
    //! It will be called when no overload is a better match.
    template<typename TAcc>
    AV2P_FN_ACC auto operator()(TAcc const& acc) const -> void
    {
        // For simplicity assume 1d thread indexing
        auto const globalThreadIdx = aV2P::getIdx<aV2P::Grid, aV2P::Threads>(acc)[0];
        if(globalThreadIdx == 0u)
            printf("Running the general kernel implementation\n");
    }

    //! Simple overload to have a special version for the CUDA accelerator
    //!
    //! We have to guard it with #ifdef as the types of alpaka accelerators are only conditionally available.
    //! Overloading for other accelerators is similar, with another template name instead of AccGpuCudaRt.
#ifdef AV2P_ACC_GPU_CUDA_ENABLED
    template<typename TDim, typename TIdx>
    AV2P_FN_ACC auto operator()(aV2P::AccGpuCudaRt<TDim, TIdx> const& acc) const -> void
    {
        // This overload is used when the kernel is run on the CUDA accelerator.
        // So inside we can use both alpaka and native CUDA directly.
        // For simplicity assume 1d thread indexing
        auto const globalThreadIdx = blockIdx.x * gridDim.x + threadIdx.x;
        if(globalThreadIdx == 0)
            printf("Running the specialization for the CUDA accelerator\n");
    }
#endif
};

// In standard projects, you typically do not execute the code with any available accelerator.
// Instead, a single accelerator is selected once from the active accelerators and the kernels are executed with the
// selected accelerator only. If you use the example as the starting point for your project, you can rename the
// example() function to main() and move the accelerator tag to the function body.
template<aV2P::concepts::Tag TAccTag>
auto example(TAccTag const&) -> int
{
    // Define the accelerator
    // For simplicity this examples always uses 1 dimensional indexing, and index type size_t
    using Acc = aV2P::TagToAcc<TAccTag, aV2P::DimInt<1>, std::size_t>;
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
    std::size_t const elementsPerGrid = 16u;
    std::size_t const elementsPerThread = 1u;
    Kernel kernel;

    aV2P::KernelCfg<Acc> const kernelCfg = {elementsPerGrid, elementsPerThread};

    // Let alpaka calculate good block and grid sizes given our full problem extent
    auto const workDiv = aV2P::getValidWorkDiv(kernelCfg, devAcc, kernel);

    // Run the kernel
    aV2P::exec<Acc>(queue, workDiv, kernel);
    aV2P::wait(queue);

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

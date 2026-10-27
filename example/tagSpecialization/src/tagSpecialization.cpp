/* Copyright 2023 Simeon Ehrig, Jan Stephan
 * SPDX-License-Identifier: ISC
 */

#include <aV2P/alpaka.hpp>
#include <aV2P/example/ExecuteForEachAccTag.hpp>

#include <iostream>

//! Different functions and kernels to illustrate specialization for a particular accelerator with tags without pre
//! processors guards.
//! The usage of tags for specialization requires the following conditions:
//!   - the code inside a specialized functions needs to be compileable independent of the enabled accelerators -> e.g.
//!     built-in function like cuda functions are not allowed
//!   - entry kernels, which are executed with aV2P::exec cannot be specialized with tags

//! Function specialization via template specialization
template<typename TTag>
std::string host_function_ver1()
{
    return "generic host function v1";
}

template<>
std::string host_function_ver1<aV2P::TagGpuCudaRt>()
{
    return "CUDA host function v1";
}

//! Function specialization via overloading
template<aV2P::concepts::Tag TTag>
std::string host_function_ver2(TTag)
{
    return "generic host function v2";
}

std::string host_function_ver2(aV2P::TagGpuCudaRt)
{
    return "CUDA host function v2";
}

//! Kernel specialization via SFINAE
//! Allows to specialize a function for more than one Acc
//! Can be also used for normal functions
template<typename TAcc, typename = void>
struct specialized_Kernel
{
    template<typename TData>
    AV2P_FN_INLINE AV2P_FN_ACC TData operator()(TAcc const& acc, TData const input)
    {
        printf("generic kernel\n");
        return aV2P::math::min(acc, 0, input);
    }
};

template<typename TAcc>
struct specialized_Kernel<
    TAcc,
    std::enable_if_t<aV2P::accMatchesTags<TAcc, aV2P::TagCpuOmp2Blocks, aV2P::TagCpuOmp2Threads>>>
{
    template<typename TData>
    AV2P_FN_INLINE AV2P_FN_ACC TData operator()(TAcc const& acc, TData const input)
    {
        printf("OpenMP kernel\n");
        return aV2P::math::min(acc, 1, input);
    }
};

struct WrapperKernel
{
    template<typename TAcc>
    AV2P_FN_ACC auto operator()(TAcc const& acc) const noexcept -> void
    {
        int const data = 42;
        printf("value of the kernel: %i\n", specialized_Kernel<TAcc>{}(acc, data));
    }
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
    using Tag = TAccTag;
    using Acc = aV2P::TagToAcc<Tag, aV2P::DimInt<1>, std::size_t>;
    std::cout << "Using alpaka accelerator: " << aV2P::getAccName<Acc>() << std::endl;

    // Call the specialized functions
    std::cout << host_function_ver1<aV2P::AccToTag<Acc>>() << std::endl;
    std::cout << host_function_ver2(aV2P::AccToTag<Acc>{}) << std::endl;

    // Defines the synchronization behavior of a queue
    using QueueProperty = aV2P::Blocking;
    using Queue = aV2P::Queue<Acc, QueueProperty>;
    // Select a device
    auto const platformAcc = aV2P::Platform<Acc>{};
    auto const devAcc = aV2P::getDevByIdx(platformAcc, 0);

    // Create a queue on the device
    Queue queue(devAcc);

    // Define the work division
    // Run only one thread independent of the acc
    aV2P::WorkDivMembers<aV2P::DimInt<1>, std::size_t> workDiv{
        static_cast<std::size_t>(1),
        static_cast<std::size_t>(1),
        static_cast<std::size_t>(1)};

    // Run the wrapper kernel, which calls the actual specialized
    // Specializing the entry kernel with tags is not possible. Therefore pre processor guards are required, see
    // kernelSpecialization example.
    aV2P::exec<Tag>(queue, workDiv, WrapperKernel{});
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

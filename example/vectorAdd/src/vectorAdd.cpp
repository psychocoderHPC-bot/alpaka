/* Copyright 2024 Benjamin Worpitz, Matthias Werner, Bernhard Manfred Gruber, Jan Stephan, Luca Ferragina,
 *                Aurora Perego, Andrea Bocci
 * SPDX-License-Identifier: ISC
 */

#include <aV2P/alpaka.hpp>
#include <aV2P/example/ExecuteForEachAccTag.hpp>

#include <chrono>
#include <iostream>
#include <random>
#include <typeinfo>

//! A vector addition kernel.
class VectorAddKernel
{
public:
    //! The kernel entry point.
    //!
    //! \tparam TAcc The accelerator environment to be executed on.
    //! \tparam TElem The matrix element type.
    //! \param acc The accelerator to be executed on.
    //! \param A The first source vector.
    //! \param B The second source vector.
    //! \param C The destination vector.
    //! \param numElements The number of elements.
    AV2P_NO_HOST_ACC_WARNING
    template<typename TAcc, typename TElem, typename TIdx>
    AV2P_FN_ACC auto operator()(
        TAcc const& acc,
        TElem const* const A,
        TElem const* const B,
        TElem* const C,
        TIdx const& numElements) const -> void
    {
        static_assert(aV2P::Dim<TAcc>::value == 1, "The VectorAddKernel expects 1-dimensional indices!");

        // The uniformElements range for loop takes care automatically of the blocks, threads and elements in the
        // kernel launch grid.
        for(auto i : aV2P::uniformElements(acc, numElements))
        {
            C[i] = A[i] + B[i];
        }
    }
};

// In standard projects, you typically do not execute the code with any available accelerator.
// Instead, a single accelerator is selected once from the active accelerators and the kernels are executed with the
// selected accelerator only. If you use the example as the starting point for your project, you can rename the
// example() function to main() and move the accelerator tag to the function body.
template<aV2P::concepts::Tag TAccTag>
auto example(TAccTag const&) -> int
{
    // Define the index domain
    // Set the number of dimensions as an integral constant. Set to 1 for 1D.
    using Dim = aV2P::DimInt<1u>;
    using Idx = std::size_t;

    // Define the accelerator
    using Acc = aV2P::TagToAcc<TAccTag, Dim, Idx>;
    using DevAcc = aV2P::Dev<Acc>;
    std::cout << "Using alpaka accelerator: " << aV2P::getAccName<Acc>() << std::endl;

    // Defines the synchronization behavior of a queue
    //
    // choose between Blocking and NonBlocking
    using QueueProperty = aV2P::Blocking;
    using QueueAcc = aV2P::Queue<Acc, QueueProperty>;

    // Select a device
    auto const platform = aV2P::Platform<Acc>{};
    auto const devAcc = aV2P::getDevByIdx(platform, 0);

    // Create a queue on the device
    QueueAcc queue(devAcc);

    // Define the work division
    Idx const numElements(123456);
    Idx const elementsPerThread(8u);
    aV2P::Vec<Dim, Idx> const extent(numElements);

    // Define the buffer element type
    using Data = std::uint32_t;

    // Get the host device for allocating memory on the host.
    using DevHost = aV2P::DevCpu;
    auto const platformHost = aV2P::PlatformCpu{};
    auto const devHost = aV2P::getDevByIdx(platformHost, 0);

    // Allocate 3 host memory buffers
    using BufHost = aV2P::Buf<DevHost, Data, Dim, Idx>;
    BufHost bufHostA(aV2P::allocBuf<Data, Idx>(devHost, extent));
    BufHost bufHostB(aV2P::allocBuf<Data, Idx>(devHost, extent));
    BufHost bufHostC(aV2P::allocBuf<Data, Idx>(devHost, extent));

    // C++14 random generator for uniformly distributed numbers in {1,..,42}
    std::random_device rd{};
    std::default_random_engine eng{rd()};
    std::uniform_int_distribution<Data> dist(1, 42);

    for(Idx i(0); i < numElements; ++i)
    {
        bufHostA[i] = dist(eng);
        bufHostB[i] = dist(eng);
        bufHostC[i] = 0;
    }

    // Allocate 3 buffers on the accelerator
    using BufAcc = aV2P::Buf<DevAcc, Data, Dim, Idx>;
    BufAcc bufAccA(aV2P::allocBuf<Data, Idx>(devAcc, extent));
    BufAcc bufAccB(aV2P::allocBuf<Data, Idx>(devAcc, extent));
    BufAcc bufAccC(aV2P::allocBuf<Data, Idx>(devAcc, extent));

    // Copy Host -> Acc
    aV2P::memcpy(queue, bufAccA, bufHostA);
    aV2P::memcpy(queue, bufAccB, bufHostB);
    aV2P::memcpy(queue, bufAccC, bufHostC);

    // Instantiate the kernel function object
    VectorAddKernel kernel;

    aV2P::KernelCfg<Acc> const kernelCfg = {extent, elementsPerThread};

    // Let alpaka calculate good block and grid sizes given our full problem extent
    auto const workDiv = aV2P::getValidWorkDiv(
        kernelCfg,
        devAcc,
        kernel,
        aV2P::getPtrNative(bufAccA),
        aV2P::getPtrNative(bufAccB),
        aV2P::getPtrNative(bufAccC),
        numElements);

    // Create the kernel execution task.
    auto const taskKernel = aV2P::createTaskKernel<Acc>(
        workDiv,
        kernel,
        std::data(bufAccA),
        std::data(bufAccB),
        std::data(bufAccC),
        numElements);

    // Enqueue the kernel execution task
    {
        // wait in case we are using an asynchronous queue to time actual kernel runtime
        aV2P::wait(queue);
        auto const beginT = std::chrono::high_resolution_clock::now();
        aV2P::enqueue(queue, taskKernel);
        // wait in case we are using an asynchronous queue to time actual kernel runtime
        aV2P::wait(queue);
        auto const endT = std::chrono::high_resolution_clock::now();
        std::cout << "Time for kernel execution: " << std::chrono::duration<double>(endT - beginT).count() << 's'
                  << std::endl;
    }

    // Copy back the result
    {
        auto beginT = std::chrono::high_resolution_clock::now();
        aV2P::memcpy(queue, bufHostC, bufAccC);
        aV2P::wait(queue);
        auto const endT = std::chrono::high_resolution_clock::now();
        std::cout << "Time for HtoD copy: " << std::chrono::duration<double>(endT - beginT).count() << 's'
                  << std::endl;
    }

    int falseResults = 0;
    static constexpr int MAX_PRINT_FALSE_RESULTS = 20;
    for(Idx i(0u); i < numElements; ++i)
    {
        Data const& val(bufHostC[i]);
        Data const correctResult(bufHostA[i] + bufHostB[i]);
        if(val != correctResult)
        {
            if(falseResults < MAX_PRINT_FALSE_RESULTS)
                std::cerr << "C[" << i << "] == " << val << " != " << correctResult << std::endl;
            ++falseResults;
        }
    }

    if(falseResults == 0)
    {
        std::cout << "Execution results correct!" << std::endl;
        return EXIT_SUCCESS;
    }
    else
    {
        std::cout << "Found " << falseResults << " false results, printed no more than " << MAX_PRINT_FALSE_RESULTS
                  << "\n"
                  << "Execution results incorrect!" << std::endl;
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

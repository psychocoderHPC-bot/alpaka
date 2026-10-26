/* Copyright 2024 Mehmet Yusufoglu, Simeon Ehrig, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/alpaka.hpp>
// Needed for running example for all backends available; one by one
#include <aV2P/example/ExecuteForEachAccTag.hpp>

#include <iostream>

//! Matrix multiplication example by using mdspan data structure

//! Some simple type traits for checking the types
//! isMdspan simply checks if a type is of type aV2P::experimental::mdspan or not
//! Primary template for is_mdspan (defaults to false)
template<typename T>
struct IsMdspan : std::false_type
{
};

//! Specialization for mdspan
template<typename... TArgs>
struct IsMdspan<aV2P::experimental::mdspan<TArgs...>> : std::true_type
{
};

template<typename T>
inline constexpr bool is_mdspan = IsMdspan<T>::value;

// Index type
using Idx = std::size_t;
// Set data type
using DataType = std::uint64_t;

/**
 * @brief Kernel for performing multiplication of two 2D matrices. Each element is computed by a different thread.
 * MdSpan data structure is used to pass the data to and from the kernel.
 */
struct MatrixMulKernel
{
    //! \tparam TAcc Accelerator type
    //! \tparam MdSpan The type of the multidimensional span (mdspan)
    //! \param acc Accelerator
    //! \param A First input matrix
    //! \param B Second input matrix
    //! \param C Output matrix where the result of A * B will be stored
    //! \param K The shared dimension between A and B
    template<typename TAcc, typename TMdSpan>
    AV2P_FN_ACC void operator()(TAcc const& acc, TMdSpan A, TMdSpan B, TMdSpan C) const
    {
        // compile time checks
        static_assert(is_mdspan<TMdSpan>, "The type TMdSpan should be an std mdspan");
        static_assert(TMdSpan::rank() == 2);

        // A is MxK and B is KxN
        auto const K = static_cast<Idx>(A.extent(1));

        auto const i = aV2P::getIdx<aV2P::Grid, aV2P::Threads>(acc)[0];
        auto const j = aV2P::getIdx<aV2P::Grid, aV2P::Threads>(acc)[1];

        if(i < C.extent(0) && j < C.extent(1))
        {
            DataType sum = 0.0f;
            for(Idx k = 0; k < K; ++k)
            {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = sum;
        }
    }
};

// initialize the matrix
template<typename TMdSpan>
inline void initializeMatrx(TMdSpan& span)
{
    auto const numColumns = span.extent(1);
    for(Idx i = 0; i < span.extent(0); ++i)
    {
        for(Idx j = 0; j < numColumns; ++j)
        {
            // fill with some data
            span(i, j) = static_cast<DataType>(i * numColumns + j);
        }
    }
}

// In standard projects, you typically do not execute the code with any available accelerator.
// Instead, a single accelerator is selected once from the active accelerators and the kernels are executed with the
// selected accelerator only. If you use the example as the starting point for your project, you can rename the
// example() function to main() and move the accelerator tag to the function body.
template<aV2P::concepts::Tag TAccTag>
auto example(TAccTag const&) -> int
{
    // Set number of dimensions (i.e 2) as a type
    using Dim = aV2P::DimInt<2>;

    // Define matrix dimensions, A is MxK and B is KxN
    Idx const M = 250;
    Idx const N = 130;
    Idx const K = 250;

    // Define device and queue
    using Acc = aV2P::TagToAcc<TAccTag, Dim, Idx>;
    using Queue = aV2P::Queue<Acc, aV2P::Blocking>;
    using Vec = aV2P::Vec<Dim, Idx>;

    auto const platformHost = aV2P::PlatformCpu{};
    auto const devHost = aV2P::getDevByIdx(platformHost, 0);
    auto const platformAcc = aV2P::Platform<Acc>{};
    auto const devAcc = aV2P::getDevByIdx(platformAcc, 0);

    Queue queue(devAcc);

    // Define the 2D extents (dimensions)
    Vec const extentA(static_cast<Idx>(M), static_cast<Idx>(K));
    Vec const extentB(static_cast<Idx>(K), static_cast<Idx>(N));
    Vec const extentC(static_cast<Idx>(M), static_cast<Idx>(N));

    // Allocate host memory
    auto bufHostA = aV2P::allocBuf<DataType, Idx>(devHost, extentA);
    auto bufHostB = aV2P::allocBuf<DataType, Idx>(devHost, extentB);
    auto bufHostC = aV2P::allocBuf<DataType, Idx>(devHost, extentC);

    // Create mdspan view for bufHostA and bufHostB using aV2P::experimental::getMdSpan to fill the host buffers
    auto mdHostA = aV2P::experimental::getMdSpan(bufHostA);
    auto mdHostB = aV2P::experimental::getMdSpan(bufHostB);

    // Initialize host matrices
    initializeMatrx(mdHostA);
    initializeMatrx(mdHostB);

    // Allocate device memory
    auto bufDevA = aV2P::allocBuf<DataType, Idx>(devAcc, extentA);
    auto bufDevB = aV2P::allocBuf<DataType, Idx>(devAcc, extentB);
    auto bufDevC = aV2P::allocBuf<DataType, Idx>(devAcc, extentC);

    // Copy data to device, use directly host buffers (not mdspans used to fill the data)
    aV2P::memcpy(queue, bufDevA, bufHostA);
    aV2P::memcpy(queue, bufDevB, bufHostB);
    aV2P::wait(queue);

    // Create mdspan views for device buffers using aV2P::experimental::getMdSpan
    auto mdDevA = aV2P::experimental::getMdSpan(bufDevA);
    auto mdDevB = aV2P::experimental::getMdSpan(bufDevB);
    auto mdDevC = aV2P::experimental::getMdSpan(bufDevC);

    MatrixMulKernel kernel;

    // Let alpaka calculate good block and grid sizes given our full problem extent
    aV2P::KernelCfg<Acc> const kernelCfg
        = {extentC, Vec::ones(), false, aV2P::GridBlockExtentSubDivRestrictions::Unrestricted};
    auto const workDiv = aV2P::getValidWorkDiv<Acc>(kernelCfg, devAcc, kernel, mdDevA, mdDevB, mdDevC);

    // Execute the kernel
    aV2P::exec<Acc>(queue, workDiv, kernel, mdDevA, mdDevB, mdDevC);

    // Copy result back to host
    aV2P::memcpy(queue, bufHostC, bufDevC);
    aV2P::wait(queue);

    // Verify the result
    bool success = true;
    auto mdHostC = aV2P::experimental::getMdSpan(bufHostC);
    for(Idx i = 0; i < M; ++i)
    {
        if(success)
        {
            for(Idx j = 0; j < N; ++j)
            {
                DataType expectedValue = 0.0f;
                for(Idx k = 0; k < K; ++k)
                {
                    expectedValue += mdHostA(i, k) * mdHostB(k, j);
                }
                if(mdHostC(i, j) != expectedValue)
                {
                    success = false;
                }
            }
        }
    }

    std::cout << "Multiplication of matrices of size " << M << "x" << K << " and " << K << "x" << N << " using mdspan "
              << (success ? "succeeded" : "failed") << "!" << std::endl;
    if(!success)
    {
        return EXIT_FAILURE;
    }

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

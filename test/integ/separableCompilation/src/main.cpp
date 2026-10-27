/* Copyright 2023 Benjamin Worpitz, Bernhard Manfred Gruber, Jan Stephan, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include "mysqrt.hpp"

#include <aV2P/test/MeasureKernelRunTime.hpp>
#include <aV2P/test/acc/TestAccs.hpp>
#include <aV2P/test/queue/Queue.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <iomanip>
#include <iostream>
#include <typeinfo>

//! A vector addition kernel.
class SqrtKernel
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

        auto const gridThreadIdx(aV2P::getIdx<aV2P::Grid, aV2P::Threads>(acc)[0]);
        auto const threadElemExtent(aV2P::getWorkDiv<aV2P::Thread, aV2P::Elems>(acc)[0]);
        auto const threadFirstElemIdx(gridThreadIdx * threadElemExtent);

        if(threadFirstElemIdx < numElements)
        {
            // Calculate the number of elements to compute in this thread.
            // The result is uniform for all but the last thread.
            auto const threadLastElemIdx(threadFirstElemIdx + threadElemExtent);
            auto const threadLastElemIdxClipped((numElements > threadLastElemIdx) ? threadLastElemIdx : numElements);

            for(TIdx i(threadFirstElemIdx); i < threadLastElemIdxClipped; ++i)
            {
                C[i] = mysqrt(A[i]) + mysqrt(B[i]);
            }
        }
    }
};

using TestAccs = aV2P::test::EnabledAccs<aV2P::DimInt<1u>, std::size_t>;

TEMPLATE_LIST_TEST_CASE("separableCompilation", "[separableCompilation]", TestAccs)
{
    using Acc = TestType;
    using Idx = aV2P::Idx<Acc>;

    using Val = float;

    using DevAcc = aV2P::Dev<Acc>;
    using PlatformAcc = aV2P::Platform<DevAcc>;
    using QueueAcc = aV2P::test::DefaultQueue<aV2P::Dev<Acc>>;

    Idx const numElements = 32;

    // Create the kernel function object.
    SqrtKernel kernel;

    // Get the host device.
    auto const platformHost = aV2P::PlatformCpu{};
    auto const devHost = aV2P::getDevByIdx(platformHost, 0);

    // Select a device to execute on.
    auto const platformAcc = PlatformAcc{};
    auto const devAcc = aV2P::getDevByIdx(platformAcc, 0);

    // Get a queue on this device.
    QueueAcc queueAcc(devAcc);

    // The data extent.
    aV2P::Vec<aV2P::DimInt<1u>, Idx> const extent(numElements);

    // Allocate host memory buffers, potentially pinned for faster copy to/from the accelerator.
    auto memBufHostA = aV2P::allocMappedBufIfSupported<Val, Idx>(devHost, platformAcc, extent);
    auto memBufHostB = aV2P::allocMappedBufIfSupported<Val, Idx>(devHost, platformAcc, extent);
    auto memBufHostC = aV2P::allocMappedBufIfSupported<Val, Idx>(devHost, platformAcc, extent);

    // Initialize the host input vectors
    for(Idx i = 0; i < numElements; ++i)
    {
        memBufHostA[i] = static_cast<Val>(rand()) / static_cast<Val>(RAND_MAX);
        memBufHostB[i] = static_cast<Val>(rand()) / static_cast<Val>(RAND_MAX);
    }

    // Allocate the buffers on the accelerator.
    auto memBufAccA = aV2P::allocBuf<Val, Idx>(devAcc, extent);
    auto memBufAccB = aV2P::allocBuf<Val, Idx>(devAcc, extent);
    auto memBufAccC = aV2P::allocBuf<Val, Idx>(devAcc, extent);

    // Copy Host -> Acc.
    aV2P::memcpy(queueAcc, memBufAccA, memBufHostA);
    aV2P::memcpy(queueAcc, memBufAccB, memBufHostB);

    // Let alpaka calculate good block and grid sizes given our full problem extent
    aV2P::KernelCfg<Acc> const kernelCfg = {extent, static_cast<Idx>(3u)};
    auto const workDiv = aV2P::getValidWorkDiv(
        kernelCfg,
        devAcc,
        kernel,
        memBufAccA.data(),
        memBufAccB.data(),
        memBufAccC.data(),
        numElements);

    std::cout << aV2P::core::demangled<decltype(kernel)> << "("
              << "accelerator: " << aV2P::getAccName<Acc>() << ", workDiv: " << workDiv
              << ", numElements:" << numElements << ")" << std::endl;

    // Create the executor task.
    auto const taskKernel = aV2P::createTaskKernel<Acc>(
        workDiv,
        kernel,
        memBufAccA.data(),
        memBufAccB.data(),
        memBufAccC.data(),
        numElements);

    // Profile the kernel execution.
    std::cout << "Execution time: " << aV2P::test::integ::measureTaskRunTimeMs(queueAcc, taskKernel) << " ms"
              << std::endl;

    // Copy back the result.
    aV2P::memcpy(queueAcc, memBufHostC, memBufAccC);
    aV2P::wait(queueAcc);

    bool resultCorrect(true);
    for(Idx i = 0; i < numElements; ++i)
    {
        auto const val = memBufHostC[i];
        auto const correctResult = std::sqrt(memBufHostA[i]) + std::sqrt(memBufHostB[i]);
        auto const absDiff = std::abs(val - correctResult);
        if(absDiff > std::numeric_limits<Val>::epsilon() * correctResult)
        {
            std::cout << std::setprecision(std::numeric_limits<Val>::digits10 + 1) << std::fixed;
            std::cout << "C[" << i << "] == " << val << " != " << correctResult << std::endl;
            resultCorrect = false;
        }
    }

    REQUIRE(true == resultCorrect);
}

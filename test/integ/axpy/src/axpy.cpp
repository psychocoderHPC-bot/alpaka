/* Copyright 2023 Axel Huebl, Benjamin Worpitz, Matthias Werner, Bernhard Manfred Gruber, Jan Stephan, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/alpaka.hpp>
#include <aV2P/test/MeasureKernelRunTime.hpp>
#include <aV2P/test/acc/TestAccs.hpp>
#include <aV2P/test/queue/Queue.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <typeinfo>

//! A vector addition kernel.
class AxpyKernel
{
public:
    //! Vector addition Y = alpha * X + Y.
    //!
    //! \tparam TAcc The type of the accelerator the kernel is executed on..
    //! \tparam TElem The matrix element type.
    //! \param acc The accelerator the kernel is executed on.
    //! \param numElements Specifies the number of elements of the vectors X and Y.
    //! \param alpha Scalar the X vector is multiplied with.
    //! \param X Vector of at least n elements.
    //! \param Y Vector of at least n elements.
    AV2P_NO_HOST_ACC_WARNING
    template<typename TAcc, typename TElem, typename TIdx>
    AV2P_FN_ACC auto operator()(
        TAcc const& acc,
        TIdx const& numElements,
        TElem const& alpha,
        TElem const* const X,
        TElem* const Y) const -> void
    {
        static_assert(aV2P::Dim<TAcc>::value == 1, "The AxpyKernel expects 1-dimensional indices!");

        auto const gridThreadIdx = aV2P::getIdx<aV2P::Grid, aV2P::Threads>(acc)[0u];
        auto const threadElemExtent = aV2P::getWorkDiv<aV2P::Thread, aV2P::Elems>(acc)[0u];
        auto const threadFirstElemIdx = gridThreadIdx * threadElemExtent;

        if(threadFirstElemIdx < numElements)
        {
            // Calculate the number of elements to compute in this thread.
            // The result is uniform for all but the last thread.
            auto const threadLastElemIdx = threadFirstElemIdx + threadElemExtent;
            auto const threadLastElemIdxClipped = (numElements > threadLastElemIdx) ? threadLastElemIdx : numElements;

            for(TIdx i(threadFirstElemIdx); i < threadLastElemIdxClipped; ++i)
            {
                Y[i] = alpha * X[i] + Y[i];
            }
        }
    }
};

using TestAccs = aV2P::test::EnabledAccs<aV2P::DimInt<1u>, std::size_t>;

TEMPLATE_LIST_TEST_CASE("axpy", "[axpy]", TestAccs)
{
    using Acc = TestType;
    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;

#ifdef AV2P_CI
    Idx const numElements = 1u << 9u;
#else
    Idx const numElements = 1u << 16u;
#endif

    using Val = float;
    using DevAcc = aV2P::Dev<Acc>;
    using QueueAcc = aV2P::test::DefaultQueue<DevAcc>;

    // Create the kernel function object.
    AxpyKernel kernel;

    // Get the host device.
    auto const platformHost = aV2P::PlatformCpu{};
    auto const devHost = aV2P::getDevByIdx(platformHost, 0);

    // Select a device to execute on.
    auto const platformAcc = aV2P::Platform<Acc>{};
    auto const devAcc = aV2P::getDevByIdx(platformAcc, 0);

    // Get a queue on this device.
    QueueAcc queue(devAcc);

    aV2P::Vec<Dim, Idx> const extent(numElements);

    // Allocate host memory buffers in pinned memory.
    auto memBufHostX = aV2P::allocMappedBufIfSupported<Val, Idx>(devHost, platformAcc, extent);
    auto memBufHostOrigY = aV2P::allocMappedBufIfSupported<Val, Idx>(devHost, platformAcc, extent);
    auto memBufHostY = aV2P::allocMappedBufIfSupported<Val, Idx>(devHost, platformAcc, extent);
    Val* const pBufHostX = std::data(memBufHostX);
    Val* const pBufHostOrigY = std::data(memBufHostOrigY);
    Val* const pBufHostY = std::data(memBufHostY);

    // random generator for uniformly distributed numbers in [0,1)
    // keep in mind, this can generate different values on different platforms
    std::random_device rd{};
    auto const seed = rd();
    std::default_random_engine eng{seed};
    std::uniform_real_distribution<Val> dist(Val{0}, Val{1});
    std::cout << "using seed: " << seed << "\n";
    // Initialize the host input vectors
    for(Idx i(0); i < numElements; ++i)
    {
        pBufHostX[i] = dist(eng);
        pBufHostOrigY[i] = dist(eng);
    }
    Val const alpha(dist(eng));

#if AV2P_DEBUG >= AV2P_DEBUG_FULL
    std::cout << __func__ << " alpha: " << alpha << std::endl;
    std::cout << __func__ << " X_host: ";
    aV2P::print(memBufHostX, std::cout);
    std::cout << std::endl;
    std::cout << __func__ << " Y_host: ";
    aV2P::print(memBufHostOrigY, std::cout);
    std::cout << std::endl;
#endif

    // Allocate the buffer on the accelerator.
    auto memBufAccX = aV2P::allocBuf<Val, Idx>(devAcc, extent);
    auto memBufAccY = aV2P::allocBuf<Val, Idx>(devAcc, extent);

    // Copy Host -> Acc.
    aV2P::memcpy(queue, memBufAccX, memBufHostX);
    aV2P::memcpy(queue, memBufAccY, memBufHostOrigY);

#if AV2P_DEBUG >= AV2P_DEBUG_FULL
    aV2P::wait(queue);

    std::cout << __func__ << " X_Dev: ";
    aV2P::print(memBufHostX, std::cout);
    std::cout << std::endl;
    std::cout << __func__ << " Y_Dev: ";
    aV2P::print(memBufHostX, std::cout);
    std::cout << std::endl;
#endif


    aV2P::KernelCfg<Acc> const kernelCfg
        = {extent, static_cast<Idx>(3u), false, aV2P::GridBlockExtentSubDivRestrictions::Unrestricted};

    // Let alpaka calculate good block and grid sizes given our full problem extent
    auto const workDiv = aV2P::getValidWorkDiv(
        kernelCfg,
        devAcc,
        kernel,
        numElements,
        alpha,
        std::data(memBufAccX),
        std::data(memBufAccY));

    std::cout << "AxpyKernel("
              << " numElements:" << numElements << ", accelerator: " << aV2P::getAccName<Acc>()
              << ", kernel: " << aV2P::core::demangled<decltype(kernel)> << ", workDiv: " << workDiv << ")"
              << std::endl;

    // Create the kernel execution task.
    auto const taskKernel = aV2P::createTaskKernel<Acc>(
        workDiv,
        kernel,
        numElements,
        alpha,
        std::data(memBufAccX),
        std::data(memBufAccY));

    // Profile the kernel execution.
    std::cout << "Execution time: " << aV2P::test::integ::measureTaskRunTimeMs(queue, taskKernel) << " ms"
              << std::endl;

    // Copy back the result.
    aV2P::memcpy(queue, memBufHostY, memBufAccY);

    // Wait for the queue to finish the memory operation.
    aV2P::wait(queue);

    bool resultCorrect(true);
    for(Idx i(0u); i < numElements; ++i)
    {
        auto const& val(pBufHostY[i]);
        auto const correctResult = alpha * pBufHostX[i] + pBufHostOrigY[i];
        auto const relDiff = std::abs((val - correctResult) / std::min(val, correctResult));
        if(relDiff > std::numeric_limits<Val>::epsilon())
        {
            std::cerr << "C[" << i << "] == " << val << " != " << correctResult << std::endl;
            resultCorrect = false;
        }
    }

    REQUIRE(resultCorrect);
}

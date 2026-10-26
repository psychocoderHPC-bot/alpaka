/* Copyright 2025 Maria Michailidi
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/mem/buf/Traits.hpp>
#include <aV2P/test/Extent.hpp>
#include <aV2P/test/acc/TestAccs.hpp>
#include <aV2P/test/mem/view/ViewTest.hpp>
#include <aV2P/test/queue/Queue.hpp>

#include <catch2/catch_message.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <numeric>
#include <type_traits>

// Kernel that adds a value to the elements of a buffer
struct ValueAddKernel
{
    template<typename Acc, typename TElem, typename TIdx>
    AV2P_FN_ACC void operator()(Acc const& acc, TElem* data, int value, TIdx numElements) const
    {
        auto const idx = aV2P::getIdx<aV2P::Grid, aV2P::Threads>(acc)[0];

        if(idx < numElements)
        {
            data[idx] += value;
        }
    }
};

TEMPLATE_LIST_TEST_CASE("memBufManagedTest", "[memBuf]", aV2P::test::TestAccs)
{
    using Acc = TestType;
    using Dev = aV2P::Dev<Acc>;
    using Queue = aV2P::test::DefaultQueue<Dev>;
    using Elem = int;
    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;

    auto const platformHost = aV2P::PlatformCpu{};
    auto const devHost = aV2P::getDevByIdx(platformHost, 0);

    auto const platformAcc = aV2P::Platform<Acc>{};
    auto const dev = aV2P::getDevByIdx(platformAcc, 0);

    INFO("Test if unified memory works in: ");
    INFO(aV2P::getName(dev));

    Queue queue(dev);

    // Configure a 1D work division large enough to cover all elements
    Idx const numElements(123456);
    Idx const elementsPerThread(1u);

    aV2P::Vec<Dim, Idx> extent = aV2P::Vec<Dim, Idx>::ones();
    extent[0] = numElements;

    aV2P::Vec<Dim, Idx> elemsPerThreadVec = aV2P::Vec<Dim, Idx>::ones();
    elemsPerThreadVec[0] = elementsPerThread;

    // Allocate unified memory associated with the host device, accessible by both host and accelerator
    auto buf = aV2P::allocManagedBuf<Elem, Idx>(devHost, platformAcc, extent);

    constexpr Elem fillVal = 42;
    auto* hostPtr = aV2P::getPtrNative(buf);
    for(Idx i = 0; i < numElements; ++i)
    {
        hostPtr[i] = fillVal;
    }

    constexpr int value = 10;

    ValueAddKernel kernel;
    aV2P::KernelCfg<Acc> const kernelCfg = {extent, elemsPerThreadVec};

    // Calculate work division
    auto const workDiv = aV2P::getValidWorkDiv(kernelCfg, dev, kernel, buf.data(), value, numElements);

    std::cout << "Testing Kernel with scalar indices with a grid of "
              << aV2P::getWorkDiv<aV2P::Grid, aV2P::Blocks>(workDiv) << " blocks x "
              << aV2P::getWorkDiv<aV2P::Block, aV2P::Threads>(workDiv) << " threads x "
              << aV2P::getWorkDiv<aV2P::Thread, aV2P::Elems>(workDiv) << " elements...\n";

    // Call of kernel to change the values of the buffer's elements
    auto const taskKernel = aV2P::createTaskKernel<Acc>(workDiv, kernel, buf.data(), value, numElements);

    aV2P::enqueue(queue, taskKernel);
    aV2P::wait(queue);

    // Verify that device writes are immediately visible on host without explicit copy
    bool passed = true;
    for(Idx i = 0; i < numElements; ++i)
    {
        if(hostPtr[i] != fillVal + value)
        {
            passed = false;
            break;
        }
    }
    CHECK(passed);
}

TEMPLATE_LIST_TEST_CASE("memBufMappedTest", "[memBuf]", aV2P::test::TestAccs)
{
    using Acc = TestType;
    using Dev = aV2P::Dev<Acc>;
    using Queue = aV2P::test::DefaultQueue<Dev>;
    using Elem = int;
    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;

    auto const platformHost = aV2P::PlatformCpu{};
    auto const devHost = aV2P::getDevByIdx(platformHost, 0);

    auto const platformAcc = aV2P::Platform<Acc>{};
    auto const dev = aV2P::getDevByIdx(platformAcc, 0);

    INFO("Test if mapped memory works in: ");
    INFO(aV2P::getName(dev));

    Queue queue(dev);

    // Configure a 1D work division large enough to cover all elements
    Idx const numElements(123456);
    Idx const elementsPerThread(1u);

    aV2P::Vec<Dim, Idx> extent = aV2P::Vec<Dim, Idx>::ones();
    extent[0] = numElements;

    aV2P::Vec<Dim, Idx> elemsPerThreadVec = aV2P::Vec<Dim, Idx>::ones();
    elemsPerThreadVec[0] = elementsPerThread;

    // Allocate pinned host memory mapped to the accelerator, accessible by both host and device
    auto buf = aV2P::allocMappedBuf<Elem, Idx>(devHost, platformAcc, extent);

    constexpr Elem fillVal = 42;
    auto* hostPtr = aV2P::getPtrNative(buf);
    for(Idx i = 0; i < numElements; ++i)
    {
        hostPtr[i] = fillVal;
    }

    constexpr int value = 10;

    ValueAddKernel kernel;
    aV2P::KernelCfg<Acc> const kernelCfg = {extent, elemsPerThreadVec};

    // Calculate work division
    auto const workDiv = aV2P::getValidWorkDiv(kernelCfg, dev, kernel, buf.data(), value, numElements);

    std::cout << "Testing Kernel with scalar indices with a grid of "
              << aV2P::getWorkDiv<aV2P::Grid, aV2P::Blocks>(workDiv) << " blocks x "
              << aV2P::getWorkDiv<aV2P::Block, aV2P::Threads>(workDiv) << " threads x "
              << aV2P::getWorkDiv<aV2P::Thread, aV2P::Elems>(workDiv) << " elements...\n";

    // Call of kernel to change the values of the buffer's elements
    auto const taskKernel = aV2P::createTaskKernel<Acc>(workDiv, kernel, buf.data(), value, numElements);

    aV2P::enqueue(queue, taskKernel);
    aV2P::wait(queue);

    // Verify that device writes to the mapped host memory are visible on the host without an explicit copy
    bool passed = true;
    for(Idx i = 0; i < numElements; ++i)
    {
        if(hostPtr[i] != fillVal + value)
        {
            passed = false;
            break;
        }
    }
    CHECK(passed);
}

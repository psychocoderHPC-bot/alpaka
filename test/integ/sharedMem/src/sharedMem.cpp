/* Copyright 2023 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera, Jan Stephan, Bernhard Manfred Gruber,
 *                Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/alpaka.hpp>
#include <aV2P/test/MeasureKernelRunTime.hpp>
#include <aV2P/test/acc/TestAccs.hpp>
#include <aV2P/test/queue/Queue.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <iostream>
#include <typeinfo>
#include <vector>

//! A kernel using atomicOp, syncBlockThreads, getDynSharedMem, getIdx, getWorkDiv and global memory to compute a
//! (useless) result. \tparam TnumUselessWork The number of useless calculations done in each kernel execution.
template<typename TnumUselessWork, typename Val>
class SharedMemKernel
{
public:
    AV2P_NO_HOST_ACC_WARNING
    template<typename TAcc>
    AV2P_FN_ACC auto operator()(TAcc const& acc, Val* const puiBlockRetVals) const -> void
    {
        using Idx = aV2P::Idx<TAcc>;

        static_assert(aV2P::Dim<TAcc>::value == 1, "The SharedMemKernel expects 1-dimensional indices!");

        // The number of threads in this block.
        Idx const blockThreadCount(aV2P::getWorkDiv<aV2P::Block, aV2P::Threads>(acc)[0u]);

        // Get the dynamically allocated shared memory.
        Val* const pBlockShared(aV2P::getDynSharedMem<Val>(acc));

        // Calculate linearized index of the thread in the block.
        Idx const blockThreadIdx1d(aV2P::getIdx<aV2P::Block, aV2P::Threads>(acc)[0u]);


        // Fill the shared block with the thread ids [1+X, 2+X, 3+X, ..., #Threads+X].
        auto sum1 = static_cast<Val>(blockThreadIdx1d + 1);
        for(Val i(0); i < static_cast<Val>(TnumUselessWork::value); ++i)
        {
            sum1 += i;
        }
        pBlockShared[blockThreadIdx1d] = sum1;


        // Synchronize all threads because now we are writing to the memory again but inverse.
        aV2P::syncBlockThreads(acc);

        // Do something useless.
        auto sum2 = static_cast<Val>(blockThreadIdx1d);
        for(Val i(0); i < static_cast<Val>(TnumUselessWork::value); ++i)
        {
            sum2 -= i;
        }
        // Add the inverse so that every cell is filled with [#Threads, #Threads, ..., #Threads].
        pBlockShared[(blockThreadCount - 1) - blockThreadIdx1d] += sum2;


        // Synchronize all threads again.
        aV2P::syncBlockThreads(acc);

        // Now add up all the cells atomically and write the result to cell 0 of the shared memory.
        if(blockThreadIdx1d > 0)
        {
            aV2P::atomicAdd(acc, &pBlockShared[0], pBlockShared[blockThreadIdx1d]);
        }


        aV2P::syncBlockThreads(acc);

        // Only master writes result to global memory.
        if(blockThreadIdx1d == 0)
        {
            // Calculate linearized block id.
            Idx const gridBlockIdx(aV2P::getIdx<aV2P::Grid, aV2P::Blocks>(acc)[0u]);

            puiBlockRetVals[gridBlockIdx] = pBlockShared[0];
        }
    }
};

namespace aV2P::trait
{
    //! The trait for getting the size of the block shared dynamic memory for a kernel.
    template<typename TnumUselessWork, typename Val, typename TAcc>
    struct BlockSharedMemDynSizeBytes<SharedMemKernel<TnumUselessWork, Val>, TAcc>
    {
        //! \return The size of the shared memory allocated for a block.
        template<typename TVec, typename... TArgs>
        AV2P_FN_HOST_ACC static auto getBlockSharedMemDynSizeBytes(
            SharedMemKernel<TnumUselessWork, Val> const& /* sharedMemKernel */,
            TVec const& blockThreadExtent,
            TVec const& /* threadElemExtent */,
            TArgs&&...) -> std::size_t
        {
            return static_cast<std::size_t>(blockThreadExtent.prod()) * sizeof(Val);
        }
    };
} // namespace aV2P::trait

using TestAccs = aV2P::test::EnabledAccs<aV2P::DimInt<1u>, std::uint32_t>;

TEMPLATE_LIST_TEST_CASE("sharedMem", "[sharedMem]", TestAccs)
{
    using Acc = TestType;
    using Idx = aV2P::Idx<Acc>;

    Idx const numElements = 1u << 16u;

    using Val = std::int32_t;
    using TnumUselessWork = std::integral_constant<Idx, 100>;

    using DevAcc = aV2P::Dev<Acc>;
    using QueueAcc = aV2P::test::DefaultQueue<DevAcc>;


    // Create the kernel function object.
    SharedMemKernel<TnumUselessWork, Val> kernel;

    // Select a device to execute on.
    auto const platformAcc = aV2P::Platform<Acc>{};
    auto const devAcc = aV2P::getDevByIdx(platformAcc, 0);

    // Get a queue on this device.
    QueueAcc queue(devAcc);


    auto blockRetValuesDummy = aV2P::allocBuf<Val, Idx>(devAcc, static_cast<Idx>(1));

    // Kernel input during the runtime of kernel will be different and is chosen to depend on workdiv.
    // Therefore, initially a  workdiv is needed to find the parameter. Therefore, in kernel bundle, we can not use the
    // real input for the buffer pointer.

    // Let alpaka calculate good block and grid sizes given our full problem extent
    aV2P::KernelCfg<Acc> const kernelCfg
        = {numElements, static_cast<Idx>(1u), false, aV2P::GridBlockExtentSubDivRestrictions::Unrestricted};
    auto const workDiv = aV2P::getValidWorkDiv(kernelCfg, devAcc, kernel, std::data(blockRetValuesDummy));

    std::cout << "SharedMemKernel("
              << " accelerator: " << aV2P::getAccName<Acc>()
              << ", kernel: " << aV2P::core::demangled<decltype(kernel)> << ", workDiv: " << workDiv << ")"
              << std::endl;

    Idx const gridBlocksCount(aV2P::getWorkDiv<aV2P::Grid, aV2P::Blocks>(workDiv)[0u]);
    Idx const blockThreadCount(aV2P::getWorkDiv<aV2P::Block, aV2P::Threads>(workDiv)[0u]);

    // An array for the return values calculated by the blocks.
    std::vector<Val> blockRetVals(static_cast<std::size_t>(gridBlocksCount));

    // Allocate accelerator buffers and copy.
    Idx const resultElemCount(gridBlocksCount);
    auto blockRetValsAcc = aV2P::allocBuf<Val, Idx>(devAcc, resultElemCount);
    aV2P::memcpy(queue, blockRetValsAcc, blockRetVals, resultElemCount);

    // Create the kernel execution task.
    auto const taskKernel = aV2P::createTaskKernel<Acc>(workDiv, kernel, std::data(blockRetValsAcc));

    // Profile the kernel execution.
    std::cout << "Execution time: " << aV2P::test::integ::measureTaskRunTimeMs(queue, taskKernel) << " ms"
              << std::endl;

    // Copy back the result.
    aV2P::memcpy(queue, blockRetVals, blockRetValsAcc, resultElemCount);

    // Wait for the queue to finish the memory operation.
    aV2P::wait(queue);

    // Assert that the results are correct.
    Val const correctResult(static_cast<Val>(blockThreadCount * blockThreadCount));

    bool resultCorrect(true);
    for(Idx i(0); i < gridBlocksCount; ++i)
    {
        auto const val(blockRetVals[static_cast<std::size_t>(i)]);
        if(val != correctResult)
        {
            std::cerr << "blockRetVals[" << i << "] == " << val << " != " << correctResult << std::endl;
            resultCorrect = false;
        }
    }

    REQUIRE(resultCorrect);
}

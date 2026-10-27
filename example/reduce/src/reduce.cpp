/* Copyright 2023 Benjamin Worpitz, Jonas Schenke, Matthias Werner, Bernhard Manfred Gruber, Jan Stephan
 * SPDX-License-Identifier: ISC
 */

#include "alpakaConfig.hpp"
#include "kernel.hpp"

#include <aV2P/alpaka.hpp>

#include <cstdlib>
#include <iostream>

// It requires support for extended lambdas when using nvcc as CUDA compiler.
// Requires sequential backend if CI is used
#if(!defined(__NVCC__) || (defined(__NVCC__) && defined(__CUDACC_EXTENDED_LAMBDA__)))                                 \
    && defined(AV2P_ACC_CPU_B_SEQ_T_SEQ_ENABLED)

// use defines of a specific accelerator from alpakaConfig.hpp
// that are defined in alpakaConfig.hpp
// - GpuCudaRt
// - CpuThreads
// - CpuOmp2Blocks
// - CpuSerial
//
using Accelerator = CpuSerial;

using Acc = Accelerator::Acc;
using Host = Accelerator::Host;
using QueueProperty = aV2P::Blocking;
using QueueAcc = aV2P::Queue<Acc, QueueProperty>;
using MaxBlockSize = Accelerator::MaxBlockSize;

//! Binary addition functor used as the device reduction operator.
struct AddFunctor
{
    template<typename T>
    AV2P_FN_ACC auto operator()(T a, T b) const -> T
    {
        return a + b;
    }
};

//! Reduces the numbers 1 to n.
//!
//! \tparam T The data type.
//! \tparam TFunc The data type of the reduction functor.
//!
//! \param devHost The host device.
//! \param devAcc The accelerator object.
//! \param queue The device queue.
//! \param n The problem size.
//! \param hostMemory The buffer containing the input data.
//! \param func The reduction function.
//!
//! Returns true if the reduction was correct and false otherwise.
template<typename T, typename DevHost, typename DevAcc, typename TFunc>
auto reduce(
    DevHost devHost,
    DevAcc devAcc,
    QueueAcc queue,
    uint64_t n,
    aV2P::Buf<DevHost, T, Dim, Idx> hostMemory,
    TFunc func) -> T
{
    static constexpr uint64_t blockSize = getMaxBlockSize<Accelerator, 256>();

    // calculate optimal block size (8 times the MP count proved to be
    // relatively near to peak performance in benchmarks)
    auto blockCount = static_cast<uint32_t>(aV2P::getAccDevProps<Acc>(devAcc).m_multiProcessorCount * 8);
    auto maxBlockCount = static_cast<uint32_t>((((n + 1) / 2) - 1) / blockSize + 1); // ceil(ceil(n/2.0)/blockSize)

    if(blockCount > maxBlockCount)
        blockCount = maxBlockCount;

    aV2P::Buf<DevAcc, T, Dim, Extent> sourceDeviceMemory = aV2P::allocBuf<T, Idx>(devAcc, n);

    aV2P::Buf<DevAcc, T, Dim, Extent> destinationDeviceMemory
        = aV2P::allocBuf<T, Idx>(devAcc, static_cast<Extent>(blockCount));

    // copy the data to the GPU
    aV2P::memcpy(queue, sourceDeviceMemory, hostMemory, n);

    // create kernels with their workdivs
    ReduceKernel<blockSize, T, TFunc> kernel1, kernel2;
    WorkDiv workDiv1{static_cast<Extent>(blockCount), static_cast<Extent>(blockSize), static_cast<Extent>(1)};
    WorkDiv workDiv2{static_cast<Extent>(1), static_cast<Extent>(blockSize), static_cast<Extent>(1)};

    // create main reduction kernel execution task
    auto const taskKernelReduceMain = aV2P::createTaskKernel<Acc>(
        workDiv1,
        kernel1,
        std::data(sourceDeviceMemory),
        std::data(destinationDeviceMemory),
        n,
        func);

    // create last block reduction kernel execution task
    auto const taskKernelReduceLastBlock = aV2P::createTaskKernel<Acc>(
        workDiv2,
        kernel2,
        std::data(destinationDeviceMemory),
        std::data(destinationDeviceMemory),
        blockCount,
        func);

    // enqueue both kernel execution tasks
    aV2P::enqueue(queue, taskKernelReduceMain);
    aV2P::enqueue(queue, taskKernelReduceLastBlock);

    //  download result from GPU
    T resultGpuHost;
    auto resultGpuDevice = aV2P::createView(devHost, &resultGpuHost, static_cast<Extent>(blockSize));

    aV2P::memcpy(queue, resultGpuDevice, destinationDeviceMemory, 1);

    return resultGpuHost;
}

auto main() -> int
{
    // select problem size
    uint64_t n = 1 << 28;

    using T = uint32_t;
    static constexpr uint64_t blockSize = getMaxBlockSize<Accelerator, 256>();

    auto const platformHost = aV2P::PlatformCpu{};
    auto const devHost = aV2P::getDevByIdx(platformHost, 0);
    auto const platformAcc = aV2P::Platform<Acc>{};
    auto const devAcc = aV2P::getDevByIdx(platformAcc, 0);
    QueueAcc queue(devAcc);

    // calculate optimal block size (8 times the MP count proved to be
    // relatively near to peak performance in benchmarks)
    uint32_t blockCount = static_cast<uint32_t>(aV2P::getAccDevProps<Acc>(devAcc).m_multiProcessorCount * 8);
    auto maxBlockCount = static_cast<uint32_t>((((n + 1) / 2) - 1) / blockSize + 1); // ceil(ceil(n/2.0)/blockSize)

    if(blockCount > maxBlockCount)
        blockCount = maxBlockCount;

    // allocate memory
    auto hostMemory = aV2P::allocBuf<T, Idx>(devHost, n);

    T* nativeHostMemory = std::data(hostMemory);

    // fill array with data
    for(uint64_t i = 0; i < n; i++)
        nativeHostMemory[i] = static_cast<T>(i + 1);

    // define the reduction function
    AddFunctor addFn;

    // reduce
    T result = reduce<T>(devHost, devAcc, queue, n, hostMemory, addFn);
    aV2P::wait(queue);

    // check result
    T expectedResult = static_cast<T>(n / 2 * (n + 1));
    if(result != expectedResult)
    {
        std::cerr << "Results don't match: " << result << " != " << expectedResult << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "Results match.\n";

    return EXIT_SUCCESS;
}

#else

int main()
{
    return EXIT_SUCCESS;
}

#endif

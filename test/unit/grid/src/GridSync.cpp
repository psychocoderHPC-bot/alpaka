/* Copyright 2026 Mykhailo Varvarin, Maria Michailidi, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/grid/Traits.hpp>
#include <aV2P/test/KernelExecutionFixture.hpp>
#include <aV2P/test/acc/TestAccs.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <type_traits>
#include <vector>

#if defined(_OPENMP)
#    include <omp.h>
#endif

class GridSyncTestKernel
{
public:
    static constexpr std::uint8_t blockThreadExtentPerDim()
    {
        return 2u;
    }

    AV2P_NO_HOST_ACC_WARNING
    template<typename TAcc, typename T>
    AV2P_FN_ACC auto operator()(TAcc const& acc, bool* success, T* array) const -> void
    {
        using Idx = aV2P::Idx<TAcc>;

        // Get the index of the current thread within the grid and the grid extent and map them to 1D.
        auto const gridThreadIdx = aV2P::getIdx<aV2P::Grid, aV2P::Threads>(acc);
        auto const gridThreadExtent = aV2P::getWorkDiv<aV2P::Grid, aV2P::Threads>(acc);
        auto const gridThreadIdx1D = aV2P::mapIdx<1u>(gridThreadIdx, gridThreadExtent)[0u];
        auto const gridThreadExtent1D = gridThreadExtent.prod();


        // Synchronize the grid multiple times, to check that the grid barrier can be reused.
        for(auto round = static_cast<Idx>(0u); round < static_cast<Idx>(3u); ++round)
        {
            // Write the thread index into the global array.
            array[gridThreadIdx1D] = static_cast<T>(gridThreadIdx1D + round * gridThreadExtent1D);

            // Synchronize the threads in the grid.
            aV2P::syncGridThreads(acc);

            // All other threads within the grid should now have written their index into the global memory.
            for(auto i = static_cast<Idx>(0u); i < gridThreadExtent1D; ++i)
            {
                AV2P_CHECK(*success, static_cast<Idx>(array[i]) == i + round * gridThreadExtent1D);
            }

            // Synchronize again, before any thread overwrites the array in the next round.
            aV2P::syncGridThreads(acc);
        }
    }
};

//! The launch configuration of a cooperative kernel: the number of threads per block, and the maximum number of blocks
//! that can be active at the same time with that block size.
template<typename TDim, typename TIdx>
struct CooperativeLaunchConfig
{
    aV2P::Vec<TDim, TIdx> threadsPerBlock;
    int maxBlocks;
};

//! Query the launch configuration of the cooperative GridSyncTestKernel. On the GPU back-ends, use the native CUDA,
//! HIP or SYCL functions to get the maximum block size for the kernel, along the innermost dimension, and the maximum
//! number of concurrent blocks with that block size. On the other back-ends, use a small block, and the largest number
//! of blocks that can be active at the same time, as given by aV2P::getMaxActiveBlocks().
template<typename TAcc>
auto nativeCooperativeLaunchConfig(aV2P::Dev<TAcc> const& dev)
    -> CooperativeLaunchConfig<aV2P::Dim<TAcc>, aV2P::Idx<TAcc>>
{
    using Tag [[maybe_unused]] = aV2P::AccToTag<TAcc>;
    using Dim = aV2P::Dim<TAcc>;
    using Idx = aV2P::Idx<TAcc>;
    using Config = CooperativeLaunchConfig<Dim, Idx>;

#if defined(AV2P_ACC_GPU_CUDA_ENABLED)
    if constexpr(std::is_same_v<Tag, aV2P::TagGpuCudaRt>)
    {
        auto const kernel = aV2P::detail::kernelName<GridSyncTestKernel, TAcc, bool*, Idx*>;
        int minGridSize = 0;
        int blockSize = 0;
        REQUIRE(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, 0) == cudaSuccess);
        int blocksPerMultiprocessor = 0;
        REQUIRE(
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerMultiprocessor, kernel, blockSize, 0)
            == cudaSuccess);
#    ifdef __CUDACC_DEBUG__
        // as in aV2P::getMaxActiveBlocks(), use at most one block per multiprocessor in device-debug mode
        blocksPerMultiprocessor = std::min(blocksPerMultiprocessor, 1);
#    endif
        int multiprocessors = 0;
        REQUIRE(
            cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, aV2P::getNativeHandle(dev))
            == cudaSuccess);
        // use the maximum block size along the innermost dimension
        auto threadsPerBlock = aV2P::Vec<Dim, Idx>::all(1);
        threadsPerBlock[Dim::value - 1] = static_cast<Idx>(blockSize);
        return Config{threadsPerBlock, blocksPerMultiprocessor * multiprocessors};
    }
#endif

#if defined(AV2P_ACC_GPU_HIP_ENABLED)
    if constexpr(std::is_same_v<Tag, aV2P::TagGpuHipRt>)
    {
        auto const kernel = aV2P::detail::kernelName<GridSyncTestKernel, TAcc, bool*, Idx*>;
        int minGridSize = 0;
        int blockSize = 0;
        REQUIRE(hipOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, 0) == hipSuccess);
        int blocksPerMultiprocessor = 0;
        REQUIRE(
            hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerMultiprocessor, kernel, blockSize, 0)
            == hipSuccess);
        int multiprocessors = 0;
        REQUIRE(
            hipDeviceGetAttribute(
                &multiprocessors,
                hipDeviceAttributeMultiprocessorCount,
                aV2P::getNativeHandle(dev))
            == hipSuccess);
        // use the maximum block size along the innermost dimension
        auto threadsPerBlock = aV2P::Vec<Dim, Idx>::all(1);
        threadsPerBlock[Dim::value - 1] = static_cast<Idx>(blockSize);
        return Config{threadsPerBlock, blocksPerMultiprocessor * multiprocessors};
    }
#endif

    // The oneAPI native occupancy query was written against the removed `detail::SyclKernel` name and
    // the old SYCL launcher helpers. It cannot run anyway: aV2P reports `m_cooperativeLaunch == false`
    // for every oneAPI executor, and `syncGridThreads` on oneAPI is the documented dependent
    // `static_assert` diagnostic (docs/aV2P/limitations.md, Section 7). The test therefore never
    // instantiates the grid-synchronization kernel for oneAPI and keeps the generic fallback below.

    // On the other back-ends, use a small block, and the largest number of blocks that can be active at the same time.
    auto const threadsPerBlock = aV2P::elementwise_min(
        aV2P::getAccDevProps<TAcc>(dev).m_blockThreadExtentMax,
        aV2P::Vec<Dim, Idx>::all(static_cast<Idx>(GridSyncTestKernel::blockThreadExtentPerDim())));
    int const maxBlocks = aV2P::getMaxActiveBlocks<TAcc>(
        dev,
        GridSyncTestKernel{},
        threadsPerBlock,
        aV2P::Vec<Dim, Idx>::all(1),
        static_cast<bool*>(nullptr),
        static_cast<Idx*>(nullptr));
    return Config{threadsPerBlock, maxBlocks};
}

TEMPLATE_LIST_TEST_CASE("synchronize", "[gridSync]", aV2P::test::TestAccs)
{
    using Acc = TestType;
    INFO(aV2P::getAccName<Acc>());
    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;

    // Select the first device available on a system, for the chosen accelerator
    auto const platformAcc = aV2P::Platform<Acc>{};
    auto const devAcc = getDevByIdx(platformAcc, 0u);

    auto const accDevProps = aV2P::trait::GetAccDevProps<Acc>::getAccDevProps(devAcc);

    // Only the host executors whose grid-wide barrier is real (the serial no-op and the OpenMP
    // team barrier) can run this test.  CUDA, HIP and oneAPI have no cooperative launch in alpaka
    // 3.0.0; `syncGridThreads` there is the documented dependent `static_assert` diagnostic
    // (docs/aV2P/limitations.md, Section 7), so the kernel must not be instantiated for them.
    constexpr bool isCooperativeCapable =
        std::is_same_v<typename Acc::Executor, alpaka::exec::CpuSerial>
#if defined(_OPENMP)
        || std::is_same_v<typename Acc::Executor, alpaka::exec::CpuOmpBlocks>
#endif
        ;

    if constexpr(isCooperativeCapable)
    {
        // The host back-ends must advertise grid synchronization.  Without this check a wrong device
        // property would silently turn the test into a no-op instead of failing.
        REQUIRE(accDevProps.m_cooperativeLaunch);
        // Use the largest block size, and the largest number of blocks that can be active at the same time.
        auto const config = nativeCooperativeLaunchConfig<Acc>(devAcc);
        auto const threadsPerBlock = config.threadsPerBlock;
        int const maxBlocks = config.maxBlocks;
        INFO("threads per block: " << threadsPerBlock << ", maximum blocks: " << maxBlocks);
        REQUIRE(threadsPerBlock.prod() > 0);
        REQUIRE(maxBlocks > 0);

        auto const elementsPerThread = aV2P::Vec<Dim, Idx>::all(1);

        GridSyncTestKernel kernel;

        // The OpenMP grid barrier requires one block per team thread; a smaller grid would leave
        // threads outside the barrier and deadlock.  The serial executor has one block by
        // construction.  Other back-ends can also test a grid smaller than the maximum.
        std::vector<int> blockCounts{maxBlocks};
        if constexpr(!std::is_same_v<typename Acc::Executor, alpaka::exec::CpuOmpBlocks>)
            blockCounts.push_back((maxBlocks + 1) / 2);

#if defined(_OPENMP)
        if constexpr(std::is_same_v<typename Acc::Executor, alpaka::exec::CpuOmpBlocks>)
        {
            // The OpenMP barrier is a grid-wide barrier only when the block count equals the team
            // size.  With a single available thread the cross-block barrier cannot be exercised;
            // fail loudly instead of silently passing a vacuous test.
            REQUIRE(maxBlocks == omp_get_max_threads());
            REQUIRE(maxBlocks > 1);
        }
#endif
        if constexpr(std::is_same_v<typename Acc::Executor, alpaka::exec::CpuSerial>)
            REQUIRE(maxBlocks == 1);

        for(int const numBlocks : blockCounts)
        {
            INFO("blocks per grid: " << numBlocks << " (maximum " << maxBlocks << ")");
            auto blocksPerGrid = aV2P::Vec<Dim, Idx>::all(1);
            blocksPerGrid[0] = static_cast<Idx>(numBlocks);

            // Allocate memory on the device, after the size of the grid is known.
            aV2P::Vec<aV2P::DimInt<1>, Idx> bufferExtent{
                blocksPerGrid.prod() * threadsPerBlock.prod() * elementsPerThread.prod()};
            auto deviceMemory = aV2P::allocBuf<Idx, Idx>(devAcc, bufferExtent);

            constexpr bool IsCooperative = true;
            aV2P::test::KernelExecutionFixture<Acc, IsCooperative> fixture(
                aV2P::WorkDivMembers<Dim, Idx>{blocksPerGrid, threadsPerBlock, elementsPerThread});

            CHECK(fixture(kernel, aV2P::getPtrNative(deviceMemory)));
        }
    }
    else
    {
        // Keep the backend visible instead of hiding it: a grid-synchronizing kernel needs a
        // cooperative launch, so these executors must keep reporting that they have none.
        REQUIRE_FALSE(accDevProps.m_cooperativeLaunch);
    }
}

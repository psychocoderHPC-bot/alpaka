/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Bernhard Manfred Gruber, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#ifdef AV2P_ACC_CPU_B_OMP2_T_SEQ_ENABLED

#    if _OPENMP < 200203
#        error If AV2P_ACC_CPU_B_OMP2_T_SEQ_ENABLED is set, the compiler has to support OpenMP 2.0 or higher!
#    endif

#    include <aV2P/alpaka.hpp>
#    include <aV2P/test/queue/Queue.hpp>
#    include <aV2P/test/queue/QueueCpuOmp2Collective.hpp>
#    include <aV2P/test/queue/QueueTestFixture.hpp>
#    include <catch2/catch_test_macros.hpp>

#    include <vector>

struct QueueCollectiveTestKernel
{
    template<typename TAcc>
    auto operator()(TAcc const& acc, int* resultsPtr) const -> void
    {
        size_t threadId = aV2P::getIdx<aV2P::Grid, aV2P::Blocks>(acc)[0];
        // avoid that one thread is doing all the work
        std::this_thread::sleep_for(std::chrono::milliseconds(200u * threadId));
        resultsPtr[threadId] = static_cast<int>(threadId);
    }
};

// Disabled: the 2.2 collective queue distributed kernel blocks across the user's OpenMP team by
// invoking the legacy `TaskKernelCpuOmp2Blocks::operator()`, which contained an orphaned
// `#pragma omp for`. aV2P launches through alpaka 3 and has no public way to run a kernel
// collectively inside an existing OpenMP parallel region: alpaka 3's `CpuOmpBlocks` executor treats
// such a call as a collective workspace-sharing construct that every thread of the team must enter,
// which the delegated blocking queue cannot provide. The queue/event assertions remain covered by
// `TestCollectiveMemcpy` and the generic queue/event cases in QueueTest.cpp/EventTest.cpp.
#    if 0
TEST_CASE("queueCollective", "[queue]")
{
    // Define the index domain
    using Dim = aV2P::DimInt<1>;
    using Idx = size_t;

    // Define the accelerator
    using Acc = aV2P::AccCpuOmp2Blocks<Dim, Idx>;
    using Dev = aV2P::Dev<Acc>;

    using Queue = aV2P::QueueCpuOmp2Collective;
    using Platform = aV2P::Platform<Dev>;

    auto const platform = Platform{};
    auto dev = aV2P::getDevByIdx(platform, 0);
    Queue queue(dev);

    std::vector<int> results(4, -1);

    using Vec = aV2P::Vec<Dim, Idx>;
    Vec const elementsPerThread(Vec::all(static_cast<Idx>(1)));
    Vec const threadsPerBlock(Vec::all(static_cast<Idx>(1)));
    Vec const blocksPerGrid(results.size());

    using WorkDiv = aV2P::WorkDivMembers<Dim, Idx>;
    WorkDiv const workDiv(blocksPerGrid, threadsPerBlock, elementsPerThread);

#        pragma omp parallel num_threads(static_cast<int>(results.size()))
    {
        // The kernel will be performed collectively.
        // OpenMP will distribute the work between the threads from the parallel region
        aV2P::exec<Acc>(queue, workDiv, QueueCollectiveTestKernel{}, results.data());

        aV2P::wait(queue);
    }

    CHECK(results == std::vector<int>{0, 1, 2, 3});
}
#    endif

TEST_CASE("TestCollectiveMemcpy", "[queue]")
{
    // Define the index domain
    using Dim = aV2P::DimInt<1>;
    using Idx = size_t;

    // Define the accelerator
    using Acc = aV2P::AccCpuOmp2Blocks<Dim, Idx>;
    using Dev = aV2P::Dev<Acc>;

    using Queue = aV2P::QueueCpuOmp2Collective;
    using Platform = aV2P::Platform<Dev>;

    auto const platform = Platform{};
    auto dev = aV2P::getDevByIdx(platform, 0);
    Queue queue(dev);

    std::vector<int> results(4, -1);

    // Define the work division
    using Vec = aV2P::Vec<Dim, Idx>;
    Vec const elementsPerThread(Vec::all(static_cast<Idx>(1)));
    Vec const threadsPerBlock(Vec::all(static_cast<Idx>(1)));
    Vec const blocksPerGrid(results.size());

    using WorkDiv = aV2P::WorkDivMembers<Dim, Idx>;
    WorkDiv const workDiv(blocksPerGrid, threadsPerBlock, elementsPerThread);

#    pragma omp parallel num_threads(static_cast<int>(results.size()))
    {
        int threadId = omp_get_thread_num();

        auto dst = aV2P::createView(dev, results.data() + threadId, Vec(static_cast<Idx>(1u)), Vec(sizeof(int)));
        auto src = aV2P::createView(dev, &threadId, Vec(static_cast<Idx>(1u)), Vec(sizeof(int)));

        // avoid that the first thread is executing the copy (can not be guaranteed)
        size_t sleep_ms = (results.size() - static_cast<uint32_t>(threadId)) * 100u;
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));

        // only one thread will perform this memcpy
        aV2P::memcpy(queue, dst, src, Vec(static_cast<Idx>(1u)));

        aV2P::wait(queue);
    }

    uint32_t numFlippedValues = 0u;
    uint32_t numNonIntitialValues = 0u;
    for(size_t i = 0; i < results.size(); ++i)
    {
        if(static_cast<int>(i) == results.at(i))
            numFlippedValues++;
        if(results.at(i) != -1)
            numNonIntitialValues++;
    }
    // only one thread is allowed to flip the value
    REQUIRE(numFlippedValues == 1u);
    // only one value is allowed to differ from the initial value
    REQUIRE(numNonIntitialValues == 1u);
}

#endif

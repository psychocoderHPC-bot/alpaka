/* Copyright 2022 Jeffrey Kelling, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/acc/Traits.hpp>
#include <aV2P/atomic/Traits.hpp>
#include <aV2P/block/shared/dyn/Traits.hpp>
#include <aV2P/block/shared/st/Traits.hpp>
#include <aV2P/block/sync/Traits.hpp>
#include <aV2P/kernel/Traits.hpp>
#include <aV2P/test/Array.hpp>
#include <aV2P/test/KernelExecutionFixture.hpp>
#include <aV2P/test/acc/TestAccs.hpp>
#include <aV2P/test/queue/Queue.hpp>
#include <aV2P/workdiv/Traits.hpp>
#include <aV2P/workdiv/WorkDivMembers.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

//! This tests checks if block-shared memory is shared correctly: only between all threads in a block.
//!
//! The check is done by each thread atomically adding 1 to a variable `shared`. Possible outcomes are:
//! * `shared < blockThreadCount`: memory is not shared between all threads of a block; usually `shared == 1`: memory
//!   is thread-private (fail)
//! * `shared > blockThreadCount`: memory is shared between blocks (fail)
//! * `shared == blockThreadCount`: memory is shared correctly (pass)
template<typename TAcc>
AV2P_FN_ACC void blockSharedMemSharingTestKernelHelper(TAcc const& acc, std::uint32_t* sums, std::uint32_t& shared)
{
    auto const blockThreadIdx = aV2P::getIdx<aV2P::Block, aV2P::Threads>(acc)[0u];
    auto const gridBlockIdx = aV2P::getIdx<aV2P::Grid, aV2P::Blocks>(acc)[0u];

    if(blockThreadIdx == 0u)
        shared = 0u;

    aV2P::syncBlockThreads(acc);

    // In the expected case we only need `hierarchy::Blocks` here, but in
    // case shared memory is shared between blocks we need atomicity
    // between blocks to get the correct result telling us so.
    aV2P::atomicAdd(acc, &shared, 1u, aV2P::hierarchy::Blocks());

    aV2P::syncBlockThreads(acc);

    if(blockThreadIdx == 0u)
        sums[gridBlockIdx] = shared;
}

template<class TAcc, class TKernel>
void BlockSharedMemSharingTest(TKernel kernel)
{
    using Dim = aV2P::Dim<TAcc>;
    using Idx = aV2P::Idx<TAcc>;
    using Vec = aV2P::Vec<Dim, Idx>;

    auto const platformAcc = aV2P::Platform<TAcc>{};
    auto const devAcc = aV2P::getDevByIdx(platformAcc, 0);

    auto const accDevProps = aV2P::getAccDevProps<TAcc>(devAcc);
    const Idx gridBlockCount = 2u;

    auto bufAcc = aV2P::allocBuf<std::uint32_t, Idx>(devAcc, gridBlockCount);

    // The maximum number of threads per block may also be limited by the kernel itself, e.g. by its stack size on
    // CUDA.
    auto const kernelFunctionAttributes = aV2P::getFunctionAttributes<TAcc>(devAcc, kernel, std::data(bufAcc));
    const Idx blockThreadCount
        = std::min(accDevProps.m_blockThreadCountMax, static_cast<Idx>(kernelFunctionAttributes.maxThreadsPerBlock));

    auto const workDiv
        = aV2P::WorkDivMembers<Dim, Idx>(Vec(gridBlockCount), Vec(blockThreadCount), Vec(static_cast<Idx>(1u)));

    auto queue = aV2P::Queue<TAcc, aV2P::Blocking>(devAcc);

    aV2P::exec<TAcc>(queue, workDiv, kernel, std::data(bufAcc));

    auto const platformHost = aV2P::PlatformCpu{};
    auto const devHost = aV2P::getDevByIdx(platformHost, 0);
    auto bufHost = aV2P::allocBuf<std::uint32_t, Idx>(devHost, gridBlockCount);

    aV2P::memcpy(queue, bufHost, bufAcc);

    auto pBufHost = std::data(bufHost);
    for(Idx a = 0u; a < gridBlockCount; ++a)
    {
        REQUIRE(pBufHost[a] == blockThreadCount);
    }
}

class BlockSharedMemStSharingTestKernel
{
public:
    AV2P_NO_HOST_ACC_WARNING
    template<typename TAcc>
    AV2P_FN_ACC void operator()(TAcc const& acc, std::uint32_t* sums) const
    {
        auto& shared = aV2P::declareSharedVar<std::uint32_t, __COUNTER__>(acc);

        blockSharedMemSharingTestKernelHelper(acc, sums, shared);
    }
};

using TestAccs = aV2P::test::EnabledAccs<aV2P::DimInt<1u>, std::uint32_t>;

TEMPLATE_LIST_TEST_CASE("blockSharedMemSt", "[blockSharedMemSharing]", TestAccs)
{
    using Acc = TestType;

    BlockSharedMemStSharingTestKernel kernel;

    BlockSharedMemSharingTest<Acc>(kernel);
}

class BlockSharedMemDynSharingTestKernel
{
public:
    AV2P_NO_HOST_ACC_WARNING
    template<typename TAcc>
    AV2P_FN_ACC void operator()(TAcc const& acc, std::uint32_t* sums) const
    {
        auto& shared = *aV2P::getDynSharedMem<std::uint32_t>(acc);

        blockSharedMemSharingTestKernelHelper(acc, sums, shared);
    }
};

namespace aV2P::trait
{
    //! The trait for getting the size of the block shared dynamic memory for a kernel.
    template<typename TAcc>
    struct BlockSharedMemDynSizeBytes<BlockSharedMemDynSharingTestKernel, TAcc>
    {
        //! \return The size of the shared memory allocated for a block.
        template<typename TVec>
        AV2P_FN_HOST_ACC static auto getBlockSharedMemDynSizeBytes(
            BlockSharedMemDynSharingTestKernel const& /* blockSharedMemDyn */,
            TVec const& /* blockThreadExtent */,
            TVec const& /* threadElemExtent */,
            std::uint32_t* /* sums */) -> std::size_t
        {
            return sizeof(std::uint32_t);
        }
    };
} // namespace aV2P::trait

TEMPLATE_LIST_TEST_CASE("blockSharedMemDyn", "[blockSharedMemSharing]", TestAccs)
{
    using Acc = TestType;

    BlockSharedMemDynSharingTestKernel kernel;

    BlockSharedMemSharingTest<Acc>(kernel);
}

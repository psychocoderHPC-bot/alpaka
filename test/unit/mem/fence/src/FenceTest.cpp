/* Copyright 2022 Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/mem/fence/Traits.hpp>
#include <aV2P/test/KernelExecutionFixture.hpp>
#include <aV2P/test/acc/TestAccs.hpp>
#include <aV2P/test/queue/Queue.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

// Trait to detect whether an accelerator supports or not multiple threads per block
template<typename TAcc>
struct IsSingleThreaded : public std::false_type
{
};

template<typename TAcc>
inline constexpr bool isSingleThreaded = IsSingleThreaded<TAcc>::value;

#ifdef AV2P_ACC_CPU_B_SEQ_T_SEQ_ENABLED
template<typename TDim, typename TIdx>
struct IsSingleThreaded<aV2P::AccCpuSerial<TDim, TIdx>> : public std::true_type
{
};
#endif // AV2P_ACC_CPU_B_SEQ_T_SEQ_ENABLED

#ifdef AV2P_ACC_CPU_B_OMP2_T_SEQ_ENABLED
template<typename TDim, typename TIdx>
struct IsSingleThreaded<aV2P::AccCpuOmp2Blocks<TDim, TIdx>> : public std::true_type
{
};
#endif // AV2P_ACC_CPU_B_OMP2_T_SEQ_ENABLED

#ifdef AV2P_ACC_CPU_B_TBB_T_SEQ_ENABLED
template<typename TDim, typename TIdx>
struct IsSingleThreaded<aV2P::AccCpuTbbBlocks<TDim, TIdx>> : public std::true_type
{
};
#endif // AV2P_ACC_CPU_B_TBB_T_SEQ_ENABLED


class DeviceFenceTestKernelWriter
{
public:
    template<typename TAcc>
    AV2P_FN_ACC auto operator()(TAcc const& acc, AV2P_DEVICE_VOLATILE int* vars) const -> void
    {
        auto const idx = aV2P::getIdx<aV2P::Grid, aV2P::Threads>(acc)[0u];

        // Use a single writer thread
        if(idx == 0)
        {
            vars[0] = 10;
            aV2P::mem_fence(acc, aV2P::mem_order::release, aV2P::memory_scope::Device{});
            vars[1] = 20;
        }
    }
};

class DeviceFenceTestKernelReader
{
public:
    template<typename TAcc>
    AV2P_FN_ACC auto operator()(TAcc const& acc, bool* success, AV2P_DEVICE_VOLATILE int* vars) const -> void
    {
        auto const idx = aV2P::getIdx<aV2P::Grid, aV2P::Threads>(acc)[0u];

        // Use a single reader thread
        if(idx == 0)
        {
            auto const b = vars[1];
            aV2P::mem_fence(acc, aV2P::mem_order::acquire, aV2P::memory_scope::Device{});
            auto const a = vars[0];

            // If the fence is working correctly, the following case can never happen
            AV2P_CHECK(*success, !(a == 1 && b == 20));
        }
    }
};

class GridFenceTestKernel
{
public:
    template<typename TAcc>
    AV2P_FN_ACC auto operator()(TAcc const& acc, bool* success, AV2P_DEVICE_VOLATILE int* vars) const -> void
    {
        auto const idx = aV2P::getIdx<aV2P::Grid, aV2P::Threads>(acc)[0u];

        // Global thread 0 is producer
        if(idx == 0)
        {
            vars[0] = 10;
            aV2P::mem_fence(acc, aV2P::mem_order::release, aV2P::memory_scope::Grid{});
            vars[1] = 20;
        }

        auto const b = vars[1];
        aV2P::mem_fence(acc, aV2P::mem_order::acquire, aV2P::memory_scope::Grid{});
        auto const a = vars[0];

        // If the fence is working correctly, the following case can never happen
        AV2P_CHECK(*success, !(a == 1 && b == 20));
    }
};

class BlockFenceTestKernel
{
public:
    template<typename TAcc>
    AV2P_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        auto const idx = aV2P::getIdx<aV2P::Block, aV2P::Threads>(acc)[0u];
        auto shared = const_cast<AV2P_DEVICE_VOLATILE int*>(aV2P::getDynSharedMem<int>(acc));

        // Initialize
        if(idx == 0)
        {
            shared[0] = 1;
            shared[1] = 2;
        }
        aV2P::syncBlockThreads(acc);

        // Local thread 0 is producer
        if(idx == 0)
        {
            shared[0] = 10;
            aV2P::mem_fence(acc, aV2P::mem_order::release, aV2P::memory_scope::Block{});
            shared[1] = 20;
        }

        auto const b = shared[1];
        aV2P::mem_fence(acc, aV2P::mem_order::acquire, aV2P::memory_scope::Block{});
        auto const a = shared[0];

        // If the fence is working correctly, the following case can never happen
        AV2P_CHECK(*success, !(a == 1 && b == 20));
    }
};

// Simply to test if compilation works
struct FenceCompileTestKernel
{
    template<typename TAcc, typename T>
    AV2P_FN_ACC auto operator()(TAcc const& acc, T* vars) const -> void
    {
        vars[0] = 1;
        aV2P::mem_fence(acc, aV2P::mem_order::relaxed, aV2P::memory_scope::Device{});
        vars[1] = 2;
        aV2P::mem_fence(acc, aV2P::mem_order::acq_rel, aV2P::memory_scope::Device{});
        vars[0] = 3;
        aV2P::mem_fence(acc, aV2P::mem_order::seq_cst, aV2P::memory_scope::Device{});
        vars[1] = 4;
        aV2P::mem_fence(acc, aV2P::memory_scope::Device{});
        vars[0] = 5;
    }
};

namespace aV2P::trait
{
    //! The trait for getting the size of the block shared dynamic memory for a kernel.
    template<typename TAcc>
    struct BlockSharedMemDynSizeBytes<BlockFenceTestKernel, TAcc>
    {
        //! \return The size of the shared memory allocated for a block.
        template<typename TVec, typename... TArgs>
        AV2P_FN_HOST_ACC static auto getBlockSharedMemDynSizeBytes(
            BlockFenceTestKernel const&,
            TVec const&,
            TVec const&,
            TArgs&&...) -> std::size_t
        {
            return 2 * sizeof(int);
        }
    };
} // namespace aV2P::trait

using TestAccs = aV2P::test::EnabledAccs<aV2P::DimInt<1u>, std::size_t>;

TEMPLATE_LIST_TEST_CASE("FenceTest", "[fence]", TestAccs)
{
    using Acc = TestType;
    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;
    using WorkDiv = aV2P::WorkDivMembers<Dim, Idx>;

    using Dev = aV2P::Dev<Acc>;
    using Platform = aV2P::Platform<Dev>;
    using Queue = aV2P::Queue<Dev, aV2P::property::NonBlocking>;

    // Fixtures with different number of blocks, threads and elements
    aV2P::Vec<Dim, Idx> const one = {1};
    aV2P::Vec<Dim, Idx> const two = {2};
    aV2P::test::KernelExecutionFixture<Acc> fixtureSingleElement{WorkDiv{one, one, one}};
    aV2P::test::KernelExecutionFixture<Acc> fixtureTwoBlocks{WorkDiv{two, one, one}};
    aV2P::test::KernelExecutionFixture<Acc> fixtureTwoElements
        = isSingleThreaded<Acc> ? aV2P::test::KernelExecutionFixture<Acc>{WorkDiv{one, one, two}}
                                : aV2P::test::KernelExecutionFixture<Acc>{WorkDiv{one, two, one}};

    auto const platformHost = aV2P::PlatformCpu{};
    auto const host = aV2P::getDevByIdx(platformHost, 0);
    auto const platformAcc = Platform{};
    auto const dev = aV2P::getDevByIdx(platformAcc, 0);
    auto queue = Queue{dev};

    auto const numElements = Idx{2ul};
    auto const extent = aV2P::Vec<Dim, Idx>{numElements};
    auto vars_host = aV2P::allocMappedBufIfSupported<int, Idx>(host, platformAcc, extent);
    auto vars_dev = aV2P::allocBuf<int, Idx>(dev, extent);
    vars_host[0] = 1;
    vars_host[1] = 2;

    // Run a single kernel, testing a memory fence in shared memory across threads in the same blocks
    BlockFenceTestKernel blockKernel;
    REQUIRE(fixtureTwoElements(blockKernel));

    // Run a single kernel, testing a memory fence in global memory across threads in different blocks
    aV2P::memcpy(queue, vars_dev, vars_host);
    aV2P::wait(queue);
    GridFenceTestKernel gridKernel;
    REQUIRE(fixtureTwoBlocks(gridKernel, vars_dev.data()));

    // Run two kernels in parallel, in two different queues on the same device, testing a memory fence
    // in global memory across threads in different grids
    aV2P::memcpy(queue, vars_dev, vars_host);
    aV2P::wait(queue);
    auto deviceKernelWriter = DeviceFenceTestKernelWriter{};
    auto deviceKernelReader = DeviceFenceTestKernelReader{};
    auto workDiv = WorkDiv{one, one, one};
    aV2P::exec<Acc>(queue, workDiv, deviceKernelWriter, vars_dev.data());
    REQUIRE(fixtureSingleElement(deviceKernelReader, vars_dev.data()));
    aV2P::wait(queue);

    // Only tests compiling and calling, no behaviour test done
    aV2P::exec<Acc>(queue, workDiv, FenceCompileTestKernel{}, vars_dev.data());
    aV2P::wait(queue);
}

/* Copyright 2026 Simone Balducci
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/mem/buf/Traits.hpp>
#include <aV2P/test/Extent.hpp>
#include <aV2P/test/acc/TestAccs.hpp>
#include <aV2P/test/mem/view/ViewTest.hpp>
#include <aV2P/test/queue/Queue.hpp>

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>

#if defined(AV2P_ACC_GPU_CUDA_ENABLED)

#    include <cub/util_allocator.cuh>

#    include <stdexcept>

namespace
{
    //! A minimal wrapper adapting cub::CachingDeviceAllocator to alpaka's aV2P::concepts::Allocator
    //! interface. The blocks are associated with the given stream, so that cub takes care of ordering their reuse
    //! with respect to the work enqueued on it.
    class AllocatorWrapper
    {
    private:
        cub::CachingDeviceAllocator* m_allocator;
        cudaStream_t m_stream;

    public:
        AllocatorWrapper(cub::CachingDeviceAllocator* allocator, cudaStream_t stream)
            : m_allocator(allocator)
            , m_stream(stream)
        {
        }

        auto allocate(std::size_t bytes, std::size_t align) -> void*
        {
            void* ptr = nullptr;
            if(m_allocator->DeviceAllocate(&ptr, bytes, m_stream) != cudaSuccess)
                throw std::runtime_error("DeviceAllocate failed");
            if(reinterpret_cast<std::uintptr_t>(ptr) % align != 0)
                throw std::runtime_error("DeviceAllocate returned unaligned pointer");
            return ptr;
        }

        auto deallocate(void* ptr) -> void
        {
            if(m_allocator->DeviceFree(ptr) != cudaSuccess)
                throw std::runtime_error("DeviceFree failed");
        }
    };

    static_assert(aV2P::concepts::Allocator<AllocatorWrapper>);
} // namespace

TEST_CASE("cachingAllocatorBufAllocTest", "[memBuf][cachingAllocator]")
{
    using Acc = aV2P::AccGpuCudaRt<aV2P::DimInt<1u>, std::size_t>;
    using Dev = aV2P::Dev<Acc>;
    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;
    using Elem = float;
    using Queue = aV2P::test::DefaultQueue<Dev>;

    auto dev = aV2P::getDevByIdx(aV2P::Platform<Acc>{}, 0u);
    Queue queue(dev);

    cub::CachingDeviceAllocator cubAllocator;
    AllocatorWrapper allocator(&cubAllocator, aV2P::getNativeHandle(queue));

    auto const extent = aV2P::test::extentBuf<Dim, Idx>;
    auto buf = aV2P::allocBuf<Elem, Idx>(dev, extent, allocator);

    auto const offset = aV2P::Vec<Dim, Idx>::zeros();
    aV2P::test::testViewImmutable<Elem>(buf, dev, extent, offset);
    aV2P::test::testViewMutable<Acc>(queue, buf);
}

TEST_CASE("cachingAllocatorAsyncBufAllocTest", "[memBuf][cachingAllocator]")
{
    using Acc = aV2P::AccGpuCudaRt<aV2P::DimInt<1u>, std::size_t>;
    using Dev = aV2P::Dev<Acc>;
    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;
    using Elem = float;
    using Queue = aV2P::test::DefaultQueue<Dev>;

    auto dev = aV2P::getDevByIdx(aV2P::Platform<Acc>{}, 0u);
    Queue queue(dev);

    cub::CachingDeviceAllocator cubAllocator;
    AllocatorWrapper allocator(&cubAllocator, aV2P::getNativeHandle(queue));

    auto const extent = aV2P::test::extentBuf<Dim, Idx>;

    STATIC_REQUIRE(aV2P::hasAsyncBufSupport<Dev, Dim>);

    auto buf = aV2P::allocAsyncBuf<Elem, Idx>(queue, extent, allocator);

    aV2P::test::testViewMutable<Acc>(queue, buf);

    aV2P::wait(queue);
    auto const offset = aV2P::Vec<Dim, Idx>::zeros();
    aV2P::test::testViewImmutable<Elem>(buf, dev, extent, offset);
}

TEST_CASE("cachingAllocatorReusesFreedMemoryTest", "[memBuf][cachingAllocator]")
{
    using Acc = aV2P::AccGpuCudaRt<aV2P::DimInt<1u>, std::size_t>;
    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;
    using Elem = float;
    using Queue = aV2P::test::DefaultQueue<aV2P::Dev<Acc>>;

    auto dev = aV2P::getDevByIdx(aV2P::Platform<Acc>{}, 0u);
    Queue queue(dev);

    cub::CachingDeviceAllocator cubAllocator;
    AllocatorWrapper allocator(&cubAllocator, aV2P::getNativeHandle(queue));

    auto const extent = aV2P::test::extentBuf<Dim, Idx>;

    void const* firstPtr = nullptr;
    {
        auto buf = aV2P::allocBuf<Elem, Idx>(dev, extent, allocator);
        firstPtr = std::data(buf);
    }

    auto buf = aV2P::allocBuf<Elem, Idx>(dev, extent, allocator);
    CHECK(firstPtr == std::data(buf));
}

#endif
